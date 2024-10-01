# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Python implementation of R-NaD (https://arxiv.org/pdf/2206.15378.pdf)."""

import enum
import functools
from typing import Any, Callable, Sequence, Tuple

import chex
import flax.linen as nn

import jax
from jax import lax
from jax import numpy as jnp
from jax import tree_util as tree
import numpy as np
import optax

from open_spiel.python import policy as policy_lib
import pyspiel

from multiprocessing import Pool
# Some handy aliases.
# Since most of these are just aliases for a "bag of tensors", the goal
# is to improve the documentation, and not to actually enforce correctness
# through pytype.
Params = chex.ArrayTree


class EntropySchedule:
  """An increasing list of steps where the regularisation network is updated.

  Example
    EntropySchedule([3, 5, 10], [2, 4, 1])
    =>   [0, 3, 6, 11, 16, 21, 26, 36]
          | 3 x2 |      5 x4     | 10 x1
  """

  def __init__(self, *, sizes: Sequence[int], repeats: Sequence[int]):
    """Constructs a schedule of entropy iterations.

    Args:
      sizes: the list of iteration sizes.
      repeats: the list, parallel to sizes, with the number of times for each
        size from `sizes` to repeat.
    """
    try:
      if len(repeats) != len(sizes):
        raise ValueError("`repeats` must be parallel to `sizes`.")
      if not sizes:
        raise ValueError("`sizes` and `repeats` must not be empty.")
      if any([(repeat <= 0) for repeat in repeats]):
        raise ValueError("All repeat values must be strictly positive")
      if repeats[-1] != 1:
        raise ValueError("The last value in `repeats` must be equal to 1, "
                         "ince the last iteration size is repeated forever.")
    except ValueError as e:
      raise ValueError(
          f"Entropy iteration schedule: repeats ({repeats}) and sizes"
          f" ({sizes})."
      ) from e

    schedule = [0]
    for size, repeat in zip(sizes, repeats):
      schedule.extend([schedule[-1] + (i + 1) * size for i in range(repeat)])

    self.schedule = np.array(schedule, dtype=np.int32)

  def __call__(self, learner_step: int) -> Tuple[float, bool]:
    """Entropy scheduling parameters for a given `learner_step`.

    Args:
      learner_step: The current learning step.

    Returns:
      alpha: The mixing weight (from [0, 1]) of the previous policy with
        the one before for computing the intrinsic reward.
      update_target_net: A boolean indicator for updating the target network
        with the current network.
    """

    # The complexity below is because at some point we might go past
    # the explicit schedule, and then we'd need to just use the last step
    # in the schedule and apply the logic of
    # ((learner_step - last_step) % last_iteration) == 0)

    # The schedule might look like this:
    # X----X-------X--X--X--X--------X
    # learner_step | might be here ^    |
    # or there     ^                    |
    # or even past the schedule         ^

    # We need to deal with two cases below.
    # Instead of going for the complicated conditional, let's just
    # compute both and then do the A * s + B * (1 - s) with s being a bool
    # selector between A and B.

    # 1. assume learner_step is past the schedule,
    #    ie schedule[-1] <= learner_step.
    last_size = self.schedule[-1] - self.schedule[-2]
    last_start = self.schedule[-1] + (
        learner_step - self.schedule[-1]) // last_size * last_size
    # 2. assume learner_step is within the schedule.
    start = jnp.amax(self.schedule * (self.schedule <= learner_step))
    finish = jnp.amin(
        self.schedule * (learner_step < self.schedule),
        initial=self.schedule[-1],
        where=(learner_step < self.schedule))
    size = finish - start

    # Now select between the two.
    beyond = (self.schedule[-1] <= learner_step)  # Are we past the schedule?
    iteration_start = (last_start * beyond + start * (1 - beyond))
    iteration_size = (last_size * beyond + size * (1 - beyond))

    update_target_net = jnp.logical_and(
        learner_step > 0,
        jnp.sum(learner_step == iteration_start + iteration_size - 1),
    )
    alpha = jnp.minimum(
        (2.0 * (learner_step - iteration_start)) / iteration_size, 1.0)

    return alpha, update_target_net  # pytype: disable=bad-return-type  # jax-types


@chex.dataclass(frozen=True)
class FineTuning:
  """Fine tuning options, aka policy post-processing.

  Even when fully trained, the resulting softmax-based policy may put
  a small probability mass on bad actions. This results in an agent
  waiting for the opponent (itself in self-play) to commit an error.

  To address that the policy is post-processed using:
  - thresholding: any action with probability smaller than self.threshold
    is simply removed from the policy.
  - discretization: the probability values are rounded to the closest
    multiple of 1/self.discretization.

  The post-processing is used on the learner, and thus must be jit-friendly.
  """
  # The learner step after which the policy post processing (aka finetuning)
  # will be enabled when learning. A strictly negative value is equivalent
  # to infinity, ie disables finetuning completely.
  from_learner_steps: int = -1
  # All policy probabilities below `threshold` are zeroed out. Thresholding
  # is disabled if this value is non-positive.
  policy_threshold: float = 0.03
  # Rounds the policy probabilities to the "closest"
  # multiple of 1/`self.discretization`.
  # Discretization is disabled for non-positive values.
  policy_discretization: int = 32

  def __call__(self, policy: chex.Array, mask: chex.Array,
               learner_steps: int) -> chex.Array:
    """A configurable fine tuning of a policy."""
    chex.assert_equal_shape((policy, mask))
    do_finetune = jnp.logical_and(self.from_learner_steps >= 0,
                                  learner_steps > self.from_learner_steps)

    return jnp.where(do_finetune, self.post_process_policy(policy, mask),
                     policy)

  def post_process_policy(
      self,
      policy: chex.Array,
      mask: chex.Array,
  ) -> chex.Array:
    """Unconditionally post process a given masked policy."""
    chex.assert_equal_shape((policy, mask))
    policy = self._threshold(policy, mask)
    policy = self._discretize(policy)
    return policy

  def _threshold(self, policy: chex.Array, mask: chex.Array) -> chex.Array:
    """Remove from the support the actions 'a' where policy(a) < threshold."""
    chex.assert_equal_shape((policy, mask))
    if self.policy_threshold <= 0:
      return policy

    mask = mask * (
        # Values over the threshold.
        (policy >= self.policy_threshold) +
        # Degenerate case is when policy is less than threshold *everywhere*.
        # In that case we just keep the policy as-is.
        (jnp.max(policy, axis=-1, keepdims=True) < self.policy_threshold))
    return mask * policy / jnp.sum(mask * policy, axis=-1, keepdims=True)

  def _discretize(self, policy: chex.Array) -> chex.Array:
    """Round all action probabilities to a multiple of 1/self.discretize."""
    if self.policy_discretization <= 0:
      return policy

    # The unbatched/single policy case:
    if len(policy.shape) == 1:
      return self._discretize_single(policy)

    # policy may be [B, A] or [T, B, A], etc. Thus add nn.BatchApply.
    dims = len(policy.shape) - 1

    # TODO(author18): avoid mixing vmap and BatchApply since the two could
    # be folded into either a single BatchApply or a sequence of vmaps, but
    # not the mix.
    vmapped = jax.vmap(self._discretize_single)
    policy = nn.BatchApply(vmapped, num_dims=dims)(policy)

    return policy

  def _discretize_single(self, mu: chex.Array) -> chex.Array:
    """A version of self._discretize but for the unbatched data."""
    # TODO(author18): try to merge _discretize and _discretize_single
    # into one function that handles both batched and unbatched cases.
    if len(mu.shape) == 2:
      mu_ = jnp.squeeze(mu, axis=0)
    else:
      mu_ = mu
    n_actions = mu_.shape[-1]
    roundup = jnp.ceil(mu_ * self.policy_discretization).astype(jnp.int32)
    result = jnp.zeros_like(mu_)
    order = jnp.argsort(-mu_)  # Indices of descending order.
    weight_left = self.policy_discretization

    def f_disc(i, order, roundup, weight_left, result):
      x = jnp.minimum(roundup[order[i]], weight_left)
      result = jax.numpy.where(weight_left >= 0, result.at[order[i]].add(x),
                               result)
      weight_left -= x
      return i + 1, order, roundup, weight_left, result

    def f_scan_scan(carry, x):
      i, order, roundup, weight_left, result = carry
      i_next, order_next, roundup_next, weight_left_next, result_next = f_disc(
          i, order, roundup, weight_left, result)
      carry_next = (i_next, order_next, roundup_next, weight_left_next,
                    result_next)
      return carry_next, x

    (_, _, _, weight_left_next, result_next), _ = jax.lax.scan(
        f_scan_scan,
        init=(jnp.asarray(0), order, roundup, weight_left, result),
        xs=None,
        length=n_actions)

    result_next = jnp.where(weight_left_next > 0,
                            result_next.at[order[0]].add(weight_left_next),
                            result_next)
    if len(mu.shape) == 2:
      result_next = jnp.expand_dims(result_next, axis=0)
    return result_next / self.policy_discretization


def _legal_policy(logits: chex.Array, legal_actions: chex.Array) -> chex.Array:
  """A soft-max policy that respects legal_actions."""
  chex.assert_equal_shape((logits, legal_actions))
  # Fiddle a bit to make sure we don't generate NaNs or Inf in the middle.
  l_min = logits.min(axis=-1, keepdims=True)
  logits = jnp.where(legal_actions, logits, l_min)
  logits -= logits.max(axis=-1, keepdims=True)
  logits *= legal_actions
  exp_logits = jnp.where(legal_actions, jnp.exp(logits),
                         0)  # Illegal actions become 0.
  exp_logits_sum = jnp.sum(exp_logits, axis=-1, keepdims=True)
  return exp_logits / exp_logits_sum


def legal_log_policy(logits: chex.Array,
                     legal_actions: chex.Array) -> chex.Array:
  """Return the log of the policy on legal action, 0 on illegal action."""
  chex.assert_equal_shape((logits, legal_actions))
  # logits_masked has illegal actions set to -inf.
  logits_masked = logits + jnp.log(legal_actions)
  max_legal_logit = logits_masked.max(axis=-1, keepdims=True)
  logits_masked = logits_masked - max_legal_logit
  # exp_logits_masked is 0 for illegal actions.
  exp_logits_masked = jnp.exp(logits_masked)

  baseline = jnp.log(jnp.sum(exp_logits_masked, axis=-1, keepdims=True))
  # Subtract baseline from logits. We do not simply return
  #     logits_masked - baseline
  # because that has -inf for illegal actions, or
  #     legal_actions * (logits_masked - baseline)
  # because that leads to 0 * -inf == nan for illegal actions.
  log_policy = jnp.multiply(legal_actions,
                            (logits - max_legal_logit - baseline))
  return log_policy


def _player_others(player_ids: chex.Array, valid: chex.Array,
                   player: int) -> chex.Array:
  """A vector of 1 for the current player and -1 for others.

  Args:
    player_ids: Tensor [...] containing player ids (0 <= player_id < N).
    valid: Tensor [...] containing whether these states are valid.
    player: The player id as int.

  Returns:
    player_other: is 1 for the current player and -1 for others [..., 1].
  """
  chex.assert_equal_shape((player_ids, valid))
  current_player_tensor = (player_ids == player).astype(jnp.int32)  # pytype: disable=attribute-error  # numpy-scalars

  res = 2 * current_player_tensor - 1
  res = res * valid
  return jnp.expand_dims(res, axis=-1)


def _policy_ratio(pi: chex.Array, mu: chex.Array, actions_oh: chex.Array,
                  valid: chex.Array) -> chex.Array:
  """Returns a ratio of policy pi/mu when selecting action a.

  By convention, this ratio is 1 on non valid states
  Args:
    pi: the policy of shape [..., A].
    mu: the sampling policy of shape [..., A].
    actions_oh: a one-hot encoding of the current actions of shape [..., A].
    valid: 0 if the state is not valid and else 1 of shape [...].

  Returns:
    pi/mu on valid states and 1 otherwise. The shape is the same
    as pi, mu or actions_oh but without the last dimension A.
  """
  chex.assert_equal_shape((pi, mu, actions_oh))
  chex.assert_shape((valid,), actions_oh.shape[:-1])

  def _select_action_prob(pi):
    return (jnp.sum(actions_oh * pi, axis=-1, keepdims=False) * valid +
            (1 - valid))

  pi_actions_prob = _select_action_prob(pi)
  mu_actions_prob = _select_action_prob(mu)
  return pi_actions_prob / mu_actions_prob


def _where(pred: chex.Array, true_data: chex.ArrayTree,
           false_data: chex.ArrayTree) -> chex.ArrayTree:
  """Similar to jax.where but treats `pred` as a broadcastable prefix."""

  def _where_one(t, f):
    chex.assert_equal_rank((t, f))
    # Expand the dimensions of pred if true_data and false_data are higher rank.
    p = jnp.reshape(pred, pred.shape + (1,) * (len(t.shape) - len(pred.shape)))
    return jnp.where(p, t, f)

  return tree.tree_map(_where_one, true_data, false_data)


def _has_played(valid: chex.Array, player_id: chex.Array,
                player: int) -> chex.Array:
  """Compute a mask of states which have a next state in the sequence."""
  chex.assert_equal_shape((valid, player_id))

  def _loop_has_played(carry, x):
    valid, player_id = x
    chex.assert_equal_shape((valid, player_id))

    our_res = jnp.ones_like(player_id)
    opp_res = carry
    reset_res = jnp.zeros_like(carry)

    our_carry = carry
    opp_carry = carry
    reset_carry = jnp.zeros_like(player_id)

    # pyformat: disable
    return _where(valid, _where((player_id == player),
                                (our_carry, our_res),
                                (opp_carry, opp_res)),
                  (reset_carry, reset_res))
    # pyformat: enable

  _, result = lax.scan(
      f=_loop_has_played,
      init=jnp.zeros_like(player_id[-1]),
      xs=(valid, player_id),
      reverse=True)
  return result


# V-Trace
#
# Custom implementation of VTrace to handle trajectories having a mix of
# different player steps. The standard rlax.vtrace can't be applied here
# out of the box because a trajectory could look like '121211221122'.


def v_trace(
    v: chex.Array,
    valid: chex.Array,
    player_id: chex.Array,
    acting_policy: chex.Array,
    merged_policy: chex.Array,
    merged_log_policy: chex.Array,
    player_others: chex.Array,
    actions_oh: chex.Array,
    reward: chex.Array,
    player: int,
    # Scalars below.
    eta: float,
    lambda_: float,
    c: float,
    rho: float,
) -> Tuple[Any, Any, Any]:
  """Custom VTrace for trajectories with a mix of different player steps."""
  gamma = 1.0

  has_played = _has_played(valid, player_id, player)

  policy_ratio = _policy_ratio(merged_policy, acting_policy, actions_oh, valid)
  inv_mu = _policy_ratio(
      jnp.ones_like(merged_policy), acting_policy, actions_oh, valid)

  eta_reg_entropy = (-eta *
                     jnp.sum(merged_policy * merged_log_policy, axis=-1) *
                     jnp.squeeze(player_others, axis=-1))
  eta_log_policy = -eta * merged_log_policy * player_others

  @chex.dataclass(frozen=True)
  class LoopVTraceCarry:
    """The carry of the v-trace scan loop."""
    reward: chex.Array
    # The cumulated reward until the end of the episode. Uncorrected (v-trace).
    # Gamma discounted and includes eta_reg_entropy.
    reward_uncorrected: chex.Array
    next_value: chex.Array
    next_v_target: chex.Array
    importance_sampling: chex.Array

  init_state_v_trace = LoopVTraceCarry(
      reward=jnp.zeros_like(reward[-1]),
      reward_uncorrected=jnp.zeros_like(reward[-1]),
      next_value=jnp.zeros_like(v[-1]),
      next_v_target=jnp.zeros_like(v[-1]),
      importance_sampling=jnp.ones_like(policy_ratio[-1]))

  def _loop_v_trace(carry: LoopVTraceCarry, x) -> Tuple[LoopVTraceCarry, Any]:
    (cs, player_id, v, reward, eta_reg_entropy, valid, inv_mu, actions_oh,
     eta_log_policy) = x

    reward_uncorrected = (
        reward + gamma * carry.reward_uncorrected + eta_reg_entropy)
    discounted_reward = reward + gamma * carry.reward

    # V-target:
    our_v_target = (
        v + jnp.expand_dims(
            jnp.minimum(rho, cs * carry.importance_sampling), axis=-1) *
        (jnp.expand_dims(reward_uncorrected, axis=-1) +
         gamma * carry.next_value - v) + lambda_ * jnp.expand_dims(
             jnp.minimum(c, cs * carry.importance_sampling), axis=-1) * gamma *
        (carry.next_v_target - carry.next_value))

    opp_v_target = jnp.zeros_like(our_v_target)
    reset_v_target = jnp.zeros_like(our_v_target)

    # Learning output:
    our_learning_output = (
        v +  # value
        eta_log_policy +  # regularisation
        actions_oh * jnp.expand_dims(inv_mu, axis=-1) *
        (jnp.expand_dims(discounted_reward, axis=-1) + gamma * jnp.expand_dims(
            carry.importance_sampling, axis=-1) * carry.next_v_target - v))

    opp_learning_output = jnp.zeros_like(our_learning_output)
    reset_learning_output = jnp.zeros_like(our_learning_output)

    # State carry:
    our_carry = LoopVTraceCarry(
        reward=jnp.zeros_like(carry.reward),
        next_value=v,
        next_v_target=our_v_target,
        reward_uncorrected=jnp.zeros_like(carry.reward_uncorrected),
        importance_sampling=jnp.ones_like(carry.importance_sampling))
    opp_carry = LoopVTraceCarry(
        reward=eta_reg_entropy + cs * discounted_reward,
        reward_uncorrected=reward_uncorrected,
        next_value=gamma * carry.next_value,
        next_v_target=gamma * carry.next_v_target,
        importance_sampling=cs * carry.importance_sampling)
    reset_carry = init_state_v_trace

    # Invalid turn: init_state_v_trace and (zero target, learning_output)
    # pyformat: disable
    return _where(valid,  # pytype: disable=bad-return-type  # numpy-scalars
                  _where((player_id == player),
                         (our_carry, (our_v_target, our_learning_output)),
                         (opp_carry, (opp_v_target, opp_learning_output))),
                  (reset_carry, (reset_v_target, reset_learning_output)))
    # pyformat: enable

  _, (v_target, learning_output) = lax.scan(
      f=_loop_v_trace,
      init=init_state_v_trace,
      xs=(policy_ratio, player_id, v, reward, eta_reg_entropy, valid, inv_mu,
          actions_oh, eta_log_policy),
      reverse=True)

  return v_target, has_played, learning_output


def get_loss_v(v_list: Sequence[chex.Array],
               v_target_list: Sequence[chex.Array],
               mask_list: Sequence[chex.Array]) -> chex.Array:
  """Define the loss function for the critic."""
  chex.assert_trees_all_equal_shapes(v_list, v_target_list)
  # v_list and v_target_list come with a degenerate trailing dimension,
  # which mask_list tensors do not have.
  chex.assert_shape(mask_list, v_list[0].shape[:-1])
  loss_v_list = []
  for (v_n, v_target, mask) in zip(v_list, v_target_list, mask_list):
    assert v_n.shape[0] == v_target.shape[0]

    loss_v = jnp.expand_dims(
        mask, axis=-1) * (v_n - lax.stop_gradient(v_target))**2
    normalization = jnp.sum(mask)
    loss_v = jnp.sum(loss_v) / (normalization + (normalization == 0.0))

    loss_v_list.append(loss_v)
  return sum(loss_v_list)

def get_l2_loss_with_mask(
              predicted: chex.Array,
              target: chex.Array,
              mask: chex.Array) -> chex.Array:
  """Define the loss function for the critic."""
  chex.assert_equal_shape((predicted, target))
  # v_list and v_target_list come with a degenerate trailing dimension,
  # which mask_list tensors do not have.
  # chex.assert_shape(mask, predicted.shape[1:-1]) # First dimension is shift in the trajectory, last is 
  # chex.assert_shape(mask_list, v_list[0].shape[:-1])
  
  loss = mask[..., jnp.newaxis] * (predicted - lax.stop_gradient(target))**2
  normalization = jnp.sum(mask)
  loss = jnp.sum(loss) / (normalization + (normalization == 0.0))
  return loss
  loss_v_list = []
  for (v_n, v_target, mask) in zip(v_list, v_target_list, mask_list):
    assert v_n.shape[0] == v_target.shape[0]

    loss_v = jnp.expand_dims(
        mask, axis=-1) * (v_n - lax.stop_gradient(v_target))**2
    normalization = jnp.sum(mask)
    loss_v = jnp.sum(loss_v) / (normalization + (normalization == 0.0))

    loss_v_list.append(loss_v)
  return sum(loss_v_list)

def apply_force_with_threshold(decision_outputs: chex.Array, force: chex.Array,
                               threshold: float,
                               threshold_center: chex.Array) -> chex.Array:
  """Apply the force with below a given threshold."""
  chex.assert_equal_shape((decision_outputs, force, threshold_center))
  can_decrease = decision_outputs - threshold_center > -threshold
  can_increase = decision_outputs - threshold_center < threshold
  force_negative = jnp.minimum(force, 0.0)
  force_positive = jnp.maximum(force, 0.0)
  clipped_force = can_decrease * force_negative + can_increase * force_positive
  return decision_outputs * lax.stop_gradient(clipped_force)


def renormalize(loss: chex.Array, mask: chex.Array) -> chex.Array:
  """The `normalization` is the number of steps over which loss is computed."""
  chex.assert_equal_shape((loss, mask))
  loss = jnp.sum(loss * mask)
  normalization = jnp.sum(mask)
  return loss / (normalization + (normalization == 0.0))


def get_loss_nerd(logit_list: Sequence[chex.Array],
                  policy_list: Sequence[chex.Array],
                  q_vr_list: Sequence[chex.Array],
                  valid: chex.Array,
                  player_ids: Sequence[chex.Array],
                  legal_actions: chex.Array,
                  importance_sampling_correction: Sequence[chex.Array],
                  clip: float = 100,
                  threshold: float = 2) -> chex.Array:
  """Define the nerd loss."""
  assert isinstance(importance_sampling_correction, list)
  loss_pi_list = []
  num_valid_actions = jnp.sum(legal_actions, axis=-1, keepdims=True)
  for k, (logit_pi, pi, q_vr, is_c) in enumerate(
      zip(logit_list, policy_list, q_vr_list, importance_sampling_correction)):
    assert logit_pi.shape[0] == q_vr.shape[0]
    # loss policy
    adv_pi = q_vr - jnp.sum(pi * q_vr, axis=-1, keepdims=True)
    adv_pi = is_c * adv_pi  # importance sampling correction
    adv_pi = jnp.clip(adv_pi, a_min=-clip, a_max=clip)
    adv_pi = lax.stop_gradient(adv_pi)

    valid_logit_sum = jnp.sum(logit_pi * legal_actions, axis=-1, keepdims=True)
    mean_logit = valid_logit_sum / num_valid_actions

    # Subtract only the mean of the valid logits
    logits = logit_pi - mean_logit

    threshold_center = jnp.zeros_like(logits)

    nerd_loss = jnp.sum(
        legal_actions *
        apply_force_with_threshold(logits, adv_pi, threshold, threshold_center),
        axis=-1)
    nerd_loss = -renormalize(nerd_loss, valid * (player_ids == k))
    loss_pi_list.append(nerd_loss)
  return sum(loss_pi_list)


@chex.dataclass(frozen=True)
class AdamConfig:
  """Adam optimizer related params."""
  b1: float = 0.0
  b2: float = 0.999
  eps: float = 10e-8


@chex.dataclass(frozen=True)
class NerdConfig:
  """Nerd related params."""
  beta: float = 2.0
  clip: float = 10_000


class StateRepresentation(str, enum.Enum):
  INFO_SET = "info_set"
  OBSERVATION = "observation"


@chex.dataclass(frozen=True)
class RNaDConfig:
  """Configuration parameters for the RNaDSolver."""
  # The game parameter string including its name and parameters.
  game_name: str
  game_params: Sequence = tuple()
  # The games longer than this value are truncated. Must be strictly positive.
  trajectory_max: int = 10

  # The content of the EnvStep.obs tensor.
  state_representation: StateRepresentation = StateRepresentation.INFO_SET

  # Network configuration.
  policy_network_layers: Sequence[int] = (256, 256)

  # The batch size to use when learning/improving parameters.
  batch_size: int = 256
  # The learning rate for `params`.
  learning_rate: float = 0.00005
  # The config related to the ADAM optimizer used for updating `params`.
  adam: AdamConfig = AdamConfig()
  # All gradients values are clipped to [-clip_gradient, clip_gradient].
  clip_gradient: float = 10_000
  # The "speed" at which `params_target` is following `params`.
  target_network_avg: float = 0.001

  # RNaD algorithm configuration.
  # Entropy schedule configuration. See EntropySchedule class documentation.
  entropy_schedule_repeats: Sequence[int] = (1,)
  entropy_schedule_size: Sequence[int] = (20_000,)
  # The weight of the reward regularisation term in RNaD.
  eta_reward_transform: float = 0.2
  nerd: NerdConfig = NerdConfig()
  c_vtrace: float = 1.0
  rho_vtrace: float = np.inf

  # Options related to fine tuning of the agent.
  finetune: FineTuning = FineTuning()

  # The seed that fully controls the randomness.
  seed: int = 42


@chex.dataclass(frozen=True)
class EnvStep:
  """Holds the tensor data representing the current game state."""
  # Indicates whether the state is a valid one or just a padding. Shape: [...]
  # The terminal state being the first one to be marked !valid.
  # All other tensors in EnvStep contain data, but only for valid timesteps.
  # Once !valid the data needs to be ignored, since it's a duplicate of
  # some other previous state.
  # The rewards is the only exception that contains reward values
  # in the terminal state, which is marked !valid.
  # TODO(author16): This is a confusion point and would need to be clarified.
  valid: chex.Array = ()  # pytype: disable=annotation-type-mismatch  # numpy-scalars
  # The single tensor representing the state observation. Shape: [..., ??]
  obs: chex.Array = ()  # pytype: disable=annotation-type-mismatch  # numpy-scalars
  # The singel tensor representing state
  state: chex.Array = () # pytype: disable=annotation-type-mismatch  # numpy-scalars
  # The legal actions mask for the current player. Shape: [..., A]
  legal: chex.Array = ()  # pytype: disable=annotation-type-mismatch  # numpy-scalars
  # The current player id as an int. Shape: [...]
  player_id: chex.Array = ()  # pytype: disable=annotation-type-mismatch  # numpy-scalars
  # The rewards of all the players. Shape: [..., P]
  rewards: chex.Array = ()  # pytype: disable=annotation-type-mismatch  # numpy-scalars


@chex.dataclass(frozen=True)
class ActorStep:
  """The actor step tensor summary."""
  # The action (as one-hot) of the current player. Shape: [..., A]
  action_oh: chex.Array = ()  # pytype: disable=annotation-type-mismatch  # numpy-scalars
  # The policy of the current player. Shape: [..., A]
  policy: chex.Array = ()  # pytype: disable=annotation-type-mismatch  # numpy-scalars
  # The rewards of all the players. Shape: [..., P]
  # Note - these are rewards obtained *after* the actor step, and thus
  # these are the same as EnvStep.rewards visible before the *next* step.
  rewards: chex.Array = ()  # pytype: disable=annotation-type-mismatch  # numpy-scalars


@chex.dataclass(frozen=True)
class TimeStep:
  """The tensor data for one game transition (env_step, actor_step)."""
  env: EnvStep = EnvStep()
  actor: ActorStep = ActorStep()


Optimizer = Callable[[Params, Params], Params]  # (params, grads) -> params


def optax_optimizer(
    params: chex.ArrayTree,
    init_and_update: optax.GradientTransformation) -> Optimizer:
  """Creates a parameterized function that represents an optimizer."""
  init_fn, update_fn = init_and_update

  @chex.dataclass
  class OptaxOptimizer:
    """A jax-friendly representation of an optimizer state with the update."""
    state: chex.Array

    def __call__(self, params: Params, grads: Params) -> Params:
      updates, self.state = update_fn(grads, self.state)  # pytype: disable=annotation-type-mismatch  # numpy-scalars
      return optax.apply_updates(params, updates)

  return OptaxOptimizer(state=init_fn(params))

class RNaDNetwork(nn.Module):
  """The RNaD network.""" 
  out_dims: int
  network_layers: Sequence[int]

  @nn.compact
  def __call__(self, env_step):
    x = env_step.obs
    for size in self.network_layers:
      x = nn.Dense(size)(x)              # create inline Flax Module submodules
      x = nn.relu(x)
    logit = nn.Dense(self.out_dims)(x)       # shape inference
    v = nn.Dense(1)(x)

    pi = _legal_policy(logit, env_step.legal)
    log_pi = legal_log_policy(logit, env_step.legal)
    return pi, v, log_pi, logit

# TODO: Shall the Legal Actions be separated? It makes sense to doit for iset, but maybe it would be better to move it to  the state network
class LegalActionsNetwork(nn.Module):
  out_dims: int # size of leagal actions
  network_layers: Sequence[int]
  
  @nn.compact
  def __call__(self, env_step):
    x = env_step.obs
    for size in self.network_layers:
      x = nn.Dense(size)(x)
      x = nn.relu(x)
    x = nn.Dense(self.out_dims)(x)
    legal_logits = nn.sigmoid(x)
    legal = jnp.where(legal_logits > 0.5, 1.0, 0.0)
    # Is it possible ot train it on this o.O ?
    return legal, legal_logits

# Network for learning next state from state and action
class DynamicsNetwork(nn.Module):
  out_dims: int # size of the next state
  network_layers: Sequence[int]
  
  @nn.compact
  def __call__(self, state, action_oh):
    x = jnp.concatenate([state, action_oh], axis=-1)
    for size in self.network_layers:
      x = nn.Dense(size)(x)
      x = nn.relu(x)
    x = nn.Dense(self.out_dims)(x)
    return x
  
  
# Outputs value for state, acting player, infoset for each player in game
class StateDetailsNetwork(nn.Module):
  players: int
  iset_size: int
  network_layers: Sequence[int]
  
  @nn.compact
  def __call__(self, env_step):
    x = env_step.state
    for size in self.network_layers:
      x = nn.Dense(size)(x)
      x = nn.relu(x)
    v = nn.Dense(1)(x)
    # Let us consider that outside of players there is also chance and terminal
    
    player_logits = nn.Dense(self.players)(x)
    acting_player = jnp.argmax(player_logits)
    # player_probs = nn.softmax(player_logits)
    
    p1_iset = nn.Dense(self.iset_size)(x)
    p2_iset = nn.Dense(self.iset_size)(x)
    
    p1_iset_logits = nn.sigmoid(p1_iset)
    p2_iset_logits = nn.sigmoid(p2_iset)
    p1_iset_rounded = jnp.where(p1_iset_logits > 0.5, 1.0, 0.0)
    p2_iset_rounded = jnp.where(p2_iset_logits > 0.5, 1.0, 0.0)
    
    return v, acting_player, player_logits, p1_iset, p2_iset, p1_iset_rounded, p2_iset_rounded

class RNaDSolver(policy_lib.Policy):
  """Implements a solver for the R-NaD Algorithm.

  See https://arxiv.org/abs/2206.15378.

  Define all networks. Derive losses & learning steps. Initialize the game
  state and algorithmic variables.
  """

  def __init__(self, config: RNaDConfig):
    self.config = config

    # Learner and actor step counters.
    self.learner_steps = 0
    self.actor_steps = 0

    self.init()

  def init(self):
    """Initialize the network and losses."""
    # The random facilities for jax and numpy.
    self._rngkey = jax.random.PRNGKey(self.config.seed)
    self._np_rng = np.random.RandomState(self.config.seed)
    # TODO(author16): serialize both above to get the fully deterministic behaviour.

    # Create a game and an example of a state.
    game_params = {}
    for n, p in self.config.game_params:
      game_params[n] = p
      
    self._game = pyspiel.load_game(self.config.game_name, game_params)
    if self._game.get_type().dynamics == pyspiel.GameType.Dynamics.SIMULTANEOUS:
        self._game = pyspiel.load_game_as_turn_based(self.config.game_name, game_params)

    self._ex_state = self._play_chance(self._game.new_initial_state())

    self.network = RNaDNetwork(self._game.num_distinct_actions(), self.config.policy_network_layers)


    # The machinery related to updating parameters/learner.
    self._entropy_schedule = EntropySchedule(
        sizes=self.config.entropy_schedule_size,
        repeats=self.config.entropy_schedule_repeats)
    self._loss_and_grad = jax.value_and_grad(self.loss, has_aux=False)
    
    self._dynamics_loss_and_grad = jax.value_and_grad(self.dynamics_loss, has_aux=False)
    
    self._legal_actions_loss_and_grad = jax.value_and_grad(self.legal_actions_loss, has_aux=False)
    
    self._state_details_loss_and_grad = jax.value_and_grad(self.state_details_loss, has_aux=False)

    # Create initial parameters.
    env_step = self._state_as_env_step(self._ex_state)
    key = self._next_rng_key()  # Make sure to use the same key for all.
    self.params = self.network.init(key, env_step)
    self.params_target = self.network.init(key, env_step)
    self.params_prev = self.network.init(key, env_step)
    self.params_prev_ = self.network.init(key, env_step)

    # Parameter optimizers.
    self.optimizer = optax_optimizer(
        self.params,
        optax.chain(
            optax.scale_by_adam(
                eps_root=0.0,
                **self.config.adam,
            ), optax.scale(-self.config.learning_rate),
            optax.clip(self.config.clip_gradient)))
    self.optimizer_target = optax_optimizer(
        self.params_target, optax.sgd(self.config.target_network_avg))
    
    if self.config.state_representation == StateRepresentation.OBSERVATION:
      obs = self._ex_state.observation_tensor()
    elif self.config.state_representation == StateRepresentation.INFO_SET:
      obs = self._ex_state.information_state_tensor()
      
      
    env_step = jax.tree_util.tree_map(lambda *xs: np.stack(xs, axis=0), env_step)
    actor_step = self.actor_step(env_step)[1]
    # TODO: Use separate config for layers
    # Initialiing MuZero stuff
    self.dynamics_network = DynamicsNetwork(len(self._ex_state.state_tensor()), self.config.policy_network_layers)
    self.legal_actions_network = LegalActionsNetwork(self._game.num_distinct_actions(), self.config.policy_network_layers)
    self.state_network = StateDetailsNetwork(2, len(obs), self.config.policy_network_layers)
    
    self.dynamics_params = self.dynamics_network.init(self._next_rng_key(), env_step.state, actor_step.action_oh)
    self.legal_actions_params = self.legal_actions_network.init(self._next_rng_key(), env_step)
    self.state_params = self.state_network.init(self._next_rng_key(), env_step)
    
    
    self.dynamics_optimizer = optax_optimizer(self.dynamics_params, optax.adam(learning_rate=self.config.learning_rate, **self.config.adam))
    self.legal_actions_optimizer = optax_optimizer(self.legal_actions_params, optax.adam(learning_rate=self.config.learning_rate, **self.config.adam))
    self.state_optimizer = optax_optimizer(self.state_params, optax.adam(learning_rate=self.config.learning_rate, **self.config.adam))
    
    # self.threads
    # self.np_rngs = [np.random.RandomState(self.config.seed + i) for i in range(self.config.batch_size)]
    # self.pool = Pool(self.config.batch_size)

  def loss(self, params: Params, params_target: Params, params_prev: Params,
           params_prev_: Params, ts: TimeStep, alpha: float,
           learner_steps: int) -> float:
    rollout = jax.vmap(self.network.apply, (None, 0), 0)
    pi, v, log_pi, logit = rollout(params, ts.env)

    policy_pprocessed = self.config.finetune(pi, ts.env.legal, learner_steps)

    _, v_target, _, _ = rollout(params_target, ts.env)
    _, _, log_pi_prev, _ = rollout(params_prev, ts.env)
    _, _, log_pi_prev_, _ = rollout(params_prev_, ts.env)
    # This line creates the reward transform log(pi(a|x)/pi_reg(a|x)).
    # For the stability reasons, reward changes smoothly between iterations.
    # The mixing between old and new reward transform is a convex combination
    # parametrised by alpha.
    log_policy_reg = log_pi - (alpha * log_pi_prev + (1 - alpha) * log_pi_prev_)

    v_target_list, has_played_list, v_trace_policy_target_list = [], [], []
    valid = jnp.where(ts.env.player_id >= 0, 1, 0)
    # valid = ts.env.valid
    for player in range(self._game.num_players()):
      reward = ts.actor.rewards[:, :, player]  # [T, B, Player]
      v_target_, has_played, policy_target_ = v_trace(
          v_target,
          valid,
          ts.env.player_id,
          ts.actor.policy,
          policy_pprocessed,
          log_policy_reg,
          _player_others(ts.env.player_id, valid, player),
          ts.actor.action_oh,
          reward,
          player,
          lambda_=1.0,
          c=self.config.c_vtrace,
          rho=self.config.rho_vtrace,
          eta=self.config.eta_reward_transform)
      v_target_list.append(v_target_)
      has_played_list.append(has_played)
      v_trace_policy_target_list.append(policy_target_)
    loss_v = get_loss_v([v] * self._game.num_players(), v_target_list,
                        has_played_list)

    is_vector = jnp.expand_dims(jnp.ones_like(ts.env.valid), axis=-1)
    importance_sampling_correction = [is_vector] * self._game.num_players()
    # Uses v-trace to define q-values for Nerd
    loss_nerd = get_loss_nerd(
        [logit] * self._game.num_players(), [pi] * self._game.num_players(),
        v_trace_policy_target_list,
        valid,
        ts.env.player_id,
        ts.env.legal,
        importance_sampling_correction,
        clip=self.config.nerd.clip,
        threshold=self.config.nerd.beta)
    return loss_v + loss_nerd  # pytype: disable=bad-return-type  # numpy-scalars

  def dynamics_loss(self, params: Params, timestep: TimeStep):
    rollout = jax.vmap(self.dynamics_network.apply, (None, 0, 0), 0)
    target_states = []
    predicted_states = []
    valids = []
    init_state = timestep.env.state
    curr_state = timestep.env.state
    actions = timestep.actor.action_oh
    valid_dynamics = jnp.triu(jnp.ones((self.config.trajectory_max, self.config.trajectory_max)))
    valid_dynamics = jnp.flip(valid_dynamics, 1)
    for i in range(self.config.trajectory_max):
      init_state = jnp.roll(init_state, shift=-1, axis=0)
      curr_state = rollout(params, curr_state, actions)
      target_states.append(init_state)
      predicted_states.append(curr_state)
      valids.append(jnp.roll(timestep.env.valid, shift=-i, axis=0))
      actions = jnp.roll(actions, shift=-1, axis=0)
    valids = jnp.stack(valids, axis=0)
    mask = valid_dynamics[..., jnp.newaxis] * valids
    # states = jnp.stack(states, axis=0)
    target_states = jnp.stack(target_states, axis=0)
    predicted_states = jnp.stack(predicted_states, axis=0)
    loss = get_l2_loss_with_mask(predicted_states, target_states, mask)
    return loss
    
  def legal_actions_loss(self, params: Params, timestep: TimeStep):
    rollout = jax.vmap(self.legal_actions_network.apply, (None, 0), 0)
    _, legal_logits = rollout(params, timestep.env)
    loss = optax.sigmoid_binary_cross_entropy(legal_logits, timestep.env.legal) * timestep.env.valid[..., jnp.newaxis]
    # loss = jnp.sum((legal - timestep.env.legal) ** 2)
    normalization = jnp.sum(timestep.env.valid)
    loss = jnp.sum(loss) / (normalization + (normalization == 0.0))
    return loss

  def state_details_loss(self, params: Params, timestep: TimeStep):
    rollout = jax.vmap(self.state_network.apply, (None, 0), 0)
    v, acting_player, player_logits, p1_iset, p2_iset, p1_iset_rounded, p2_iset_rounded = rollout(params, timestep.env)
    player_id = timestep.env.player_id + 2 # chance is -1 terminal -4, so after this the terminal should be -2.
    player_id = jnp.where(player_id < 0, 0, player_id) 
    normalization = jnp.sum(timestep.env.valid)
    
    player_loss = optax.softmax_cross_entropy_with_integer_labels(player_logits, player_id) * timestep.env.valid
    player_loss = jnp.sum(player_loss) / (normalization + (normalization == 0.0))
    
    v_loss =  ((jnp.squeeze(v) - timestep.actor.rewards[..., 0]) ** 2) * timestep.env.valid
    v_loss = jnp.sum(v_loss) / (normalization + (normalization == 0.0))
    
    p1_mask = jnp.where(player_id == 2, 1, 0) * timestep.env.valid
    p2_mask = jnp.where(player_id == 3, 1, 0) * timestep.env.valid
    p1_iset_loss = optax.sigmoid_binary_cross_entropy(p1_iset, timestep.env.obs) * p1_mask[..., jnp.newaxis]
    p2_iset_loss = optax.sigmoid_binary_cross_entropy(p2_iset, timestep.env.obs) * p2_mask[..., jnp.newaxis]
    
    p1_normalization = jnp.sum(p1_mask)
    p2_normalization = jnp.sum(p2_mask)
    
    p1_iset_loss = jnp.sum(p1_iset_loss) / (p1_normalization + (p1_normalization == 0.0))
    p2_iset_loss = jnp.sum(p2_iset_loss) / (p2_normalization + (p2_normalization == 0.0))
    
    return player_loss + v_loss + p1_iset_loss + p2_iset_loss

  @functools.partial(jax.jit, static_argnums=(0,))
  def update_mu_zero_parameters(
      self,
      legal_actions_params: Params,
      dynamics_params: Params,
      state_params: Params,
      legal_actions_optimizer: Optimizer,
      dynamics_optimizer: Optimizer,
      state_optimizer: Optimizer,
      timestep: TimeStep
      ):
    
    dynamics_loss, dynamics_grad = self._dynamics_loss_and_grad(dynamics_params, timestep)
    legal_actions_loss, legal_actions_grad = self._legal_actions_loss_and_grad(legal_actions_params, timestep)
    state_loss, state_grad = self._state_details_loss_and_grad(state_params, timestep)
    
    dynamics_params = dynamics_optimizer(dynamics_params, dynamics_grad)
    legal_actions_params = legal_actions_optimizer(legal_actions_params, legal_actions_grad)
    state_params = state_optimizer(state_params, state_grad)
    
    logs = {"dynamics_loss": dynamics_loss, "legal_actions_loss": legal_actions_loss, "state_loss": state_loss}
    return (legal_actions_params, dynamics_params, state_params, legal_actions_optimizer, dynamics_optimizer, state_optimizer), logs

  @functools.partial(jax.jit, static_argnums=(0,))
  def update_parameters(
      self,
      params: Params,
      params_target: Params,
      params_prev: Params,
      params_prev_: Params,
      optimizer: Optimizer,
      optimizer_target: Optimizer,
      timestep: TimeStep,
      alpha: float,
      learner_steps: int,
      update_target_net: bool):
    """A jitted pure-functional part of the `step`."""
    loss_val, grad = self._loss_and_grad(params, params_target, params_prev,
                                         params_prev_, timestep, alpha,
                                         learner_steps)
    # Update `params`` using the computed gradient.
    params = optimizer(params, grad)
    # Update `params_target` towards `params`.
    params_target = optimizer_target(
        params_target, tree.tree_map(lambda a, b: a - b, params_target, params))

    # Rolls forward the prev and prev_ params if update_target_net is 1.
    # pyformat: disable
    params_prev, params_prev_ = jax.lax.cond(
        update_target_net,
        lambda: (params_target, params_prev),
        lambda: (params_prev, params_prev_))
    # pyformat: enable

    logs = {
        "loss": loss_val,
    }
    return (params, params_target, params_prev, params_prev_, optimizer,
            optimizer_target), logs

  def __getstate__(self):
    """To serialize the agent."""
    return dict(
        # RNaD config.
        config=self.config,

        # Learner and actor step counters.
        learner_steps=self.learner_steps,
        actor_steps=self.actor_steps,

        # The randomness keys.
        np_rng=self._np_rng.get_state(),
        rngkey=self._rngkey,

        # Network params.
        params=self.params,
        params_target=self.params_target,
        params_prev=self.params_prev,
        params_prev_=self.params_prev_,
        # Optimizer state.
        optimizer=self.optimizer.state,  # pytype: disable=attribute-error  # always-use-return-annotations
        optimizer_target=self.optimizer_target.state,  # pytype: disable=attribute-error  # always-use-return-annotations
        
        legal_actions_params=self.legal_actions_params,
        dynamics_params=self.dynamics_params,
        state_params=self.state_params,
        legal_actions_optimizer=self.legal_actions_optimizer.state,
        dynamics_optimizer=self.dynamics_optimizer.state,
        state_optimizer=self.state_optimizer.state
    )

  def __setstate__(self, state):
    """To deserialize the agent."""
    # RNaD config.
    self.config = state["config"]

    self.init()

    # Learner and actor step counters.
    self.learner_steps = state["learner_steps"]
    self.actor_steps = state["actor_steps"]

    # The randomness keys.
    self._np_rng.set_state(state["np_rng"])
    self._rngkey = state["rngkey"]

    # Network params.
    self.params = state["params"]
    self.params_target = state["params_target"]
    self.params_prev = state["params_prev"]
    self.params_prev_ = state["params_prev_"]
    # Optimizer state.
    self.optimizer.state = state["optimizer"]
    self.optimizer_target.state = state["optimizer_target"]
    
    self.legal_actions_params = state["legal_actions_params"]
    self.dynamics_params = state["dynamics_params"]
    self.state_params = state["state_params"]
    self.legal_actions_optimizer.state = state["legal_actions_optimizer"]
    self.dynamics_optimizer.state = state["dynamics_optimizer"]
    self.state_optimizer.state = state["state_optimizer"]

  def step(self):
    """One step of the algorithm, that plays the game and improves params."""
    timestep = self.collect_batch_trajectory()
    alpha, update_target_net = self._entropy_schedule(self.learner_steps)
    (self.params, self.params_target, self.params_prev, self.params_prev_,
     self.optimizer, self.optimizer_target), logs = self.update_parameters(
         self.params, self.params_target, self.params_prev, self.params_prev_,
         self.optimizer, self.optimizer_target, timestep, alpha,
         self.learner_steps, update_target_net)
    (self.legal_actions_params, self.dynamics_params, self.state_prarams, self.legal_actions_optimizer, self.dynamics_optimizer, self.state_optimizer), dynamics_logs = self.update_mu_zero_parameters(self.legal_actions_params, self.dynamics_params, self.state_params, self.legal_actions_optimizer, self.dynamics_optimizer, self.state_optimizer, timestep)
    
    self.learner_steps += 1
    logs.update(dynamics_logs)
    logs.update({
        "actor_steps": self.actor_steps,
        "learner_steps": self.learner_steps,
    })
    return logs

  def _next_rng_key(self) -> chex.PRNGKey:
    """Get the next rng subkey from class rngkey.

    Must *not* be called from under a jitted function!

    Returns:
      A fresh rng_key.
    """
    self._rngkey, subkey = jax.random.split(self._rngkey)
    return subkey

  def _next_rng_keys(self, keys) -> list[chex.PRNGKey]:
    self._rngkey, *subkeys = jax.random.split(self._rngkey, keys + 1)
    return subkeys

  def _state_as_env_step(self, state: pyspiel.State) -> EnvStep:
    # A terminal state must be communicated to players, however since
    # it's a terminal state things like the state_representation or
    # the set of legal actions are meaningless and only needed
    # for the sake of creating well a defined trajectory tensor.
    # Therefore the code below:
    # - extracts the rewards
    # - if the state is terminal, uses a dummy other state for other fields.
    rewards = np.array(state.returns(), dtype=np.float64)

    # we need state tensors even for terminals
    state_tensor = state.state_tensor()

    if self.config.state_representation == StateRepresentation.OBSERVATION:
      obs = state.observation_tensor()
    elif self.config.state_representation == StateRepresentation.INFO_SET:
      obs = state.information_state_tensor()
    else:
      raise ValueError(
          f"Invalid StateRepresentation: {self.config.state_representation}.")

    # TODO: Split valid into terminal and valid, since in terminal we still need to train the networks, in invalid we do not.
    valid = not state.is_terminal()
      
    # TODO(author16): clarify the story around rewards and valid.
    return EnvStep(
        obs=np.array(obs, dtype=np.float64),
        state = np.array(state_tensor, dtype=np.float64),
        legal=np.array(state.legal_actions_mask(), dtype=np.int8),
        player_id=np.array(state.current_player(), dtype=np.float64),
        valid=np.array(valid, dtype=np.float64),
        rewards=rewards)

  def action_probabilities(self,
                           state: pyspiel.State,
                           player_id: Any = None):
    """Returns action probabilities dict for a single batch."""
    env_step = self._batch_of_states_as_env_step([state])
    probs = self._network_jit_apply_and_post_process(
        self.params_target, env_step)
    probs = jax.device_get(probs[0])  # Squeeze out the 1-element batch.
    return {
        action: probs[action]
        for action, valid in enumerate(jax.device_get(env_step.legal[0]))
        if valid
    }

  @functools.partial(jax.jit, static_argnums=(0,))
  def _network_jit_apply_and_post_process(
      self, params: Params, env_step: EnvStep) -> chex.Array:
    pi, _, _, _ = self.network.apply(params, env_step)
    pi = self.config.finetune.post_process_policy(pi, env_step.legal)
    return pi

  @functools.partial(jax.jit, static_argnums=(0,))
  def _network_jit_apply(self, params: Params, env_step: EnvStep) -> chex.Array:
    pi, _, _, _ = self.network.apply(params, env_step)
    # pi = jnp.where(jnp.sum(env_step.legal, -1, keepdims=True) == 0, 1, pi) # This may happen in terminal states
    return pi

  # Expects env_step to be [B, ...], meaning it has to have leading dimension at least 1
  @functools.partial(jax.jit, static_argnums=(0, 1))
  def _jit_sample_batch_action(self, distinct_actions: int, pi: chex.Array, rngkeys: chex.Array):
    def choice_wrapper(key, probs, amount_actions):
      return jax.random.choice(key, amount_actions, p=probs)
    
    vectorized_choice = jax.vmap(choice_wrapper, (0, 0, None), 0)
    action = vectorized_choice(rngkeys, pi, distinct_actions)
    action_oh = jax.nn.one_hot(action, distinct_actions)
    return action, action_oh

  @functools.partial(jax.jit, static_argnums=(0, 1))
  def _network_jit_apply_and_sample(self, distinct_actions: int, params: Params, env_step: EnvStep, rngkeys: chex.Array) -> chex.Array:
    pi = self._network_jit_apply(params, env_step)
    action, action_oh = self._jit_sample_batch_action(distinct_actions, pi, rngkeys)
    return pi, action, action_oh

  def actor_step(self, env_step: EnvStep):
    # Shape[0] should be batch
    keys = self._next_rng_keys(env_step.obs.shape[0])
    keys = np.asarray(keys)
    pi, action, action_oh = self._network_jit_apply_and_sample(self._game.num_distinct_actions(), self.params, env_step, keys)
    pi = np.asarray(pi, dtype=np.float32)
    action = np.asarray(action, dtype=np.int32)
    action_oh = np.asarray(action_oh, dtype=np.float32)

    actor_step = ActorStep(policy=pi, action_oh=action_oh, rewards=())

    return action, actor_step 
  
  
  def collect_batch_trajectory(self) -> TimeStep:
    states = [
        self._play_chance(self._game.new_initial_state())
        for _ in range(self.config.batch_size)
    ]
    timesteps = []

    env_step = self._batch_of_states_as_env_step(states)
    for _ in range(self.config.trajectory_max):
      prev_env_step = env_step
      a, actor_step = self.actor_step(env_step)

      succseful, states = self._batch_of_states_apply_action(states, a)
      if not succseful:
        break
      env_step = self._batch_of_states_as_env_step(states)
      # We say that invalid states are only the padding states after the terminal state in previous step.
      env_step = EnvStep(
          obs=env_step.obs,
          state=env_step.state,
          legal=env_step.legal,
          player_id=env_step.player_id,
          valid=np.where(prev_env_step.player_id == -4, 0, 1),
          rewards=env_step.rewards
      )
      timesteps.append(
          TimeStep(
              # env=EnvStep(
              #   obs=prev_env_step.obs,
              #   state=prev_env_step.state,
              #   legal=prev_env_step.legal,
              #   player_id=prev_env_step.player_id,
              #   valid=prev_env_step.valid,
              #   rewards=prev_env_step.rewards
              # ),
              env=prev_env_step,
              actor=ActorStep(
                  action_oh=actor_step.action_oh,
                  policy=actor_step.policy,
                  rewards=env_step.rewards),
          ))
    if not succseful:
      print("Redoing batch: ", self.learner_steps, flush=True)
      return self.collect_batch_trajectory()
    # Concatenate all the timesteps together to form a single rollout [T, B, ..]
    return jax.tree_util.tree_map(lambda *xs: np.stack(xs, axis=0), *timesteps)
 
  
  def _prepare_env_step(self, state: pyspiel.State) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
    rewards = state.returns()
    
    iset_state = state
    valid = not state.is_terminal()
    if not valid:
      iset_state = self._ex_state
    
    if self.config.state_representation == StateRepresentation.OBSERVATION:
      obs = iset_state.observation_tensor()
    elif self.config.state_representation == StateRepresentation.INFO_SET:
      obs = iset_state.information_state_tensor()
    else:
      raise ValueError(
          f"Invalid StateRepresentation: {self.config.state_representation}.")
    
    
      
    return rewards, iset_state.legal_actions_mask(), state.current_player(), valid, obs, state.state_tensor()

  def _batch_of_states_as_env_step(self,
                                   states: Sequence[pyspiel.State]) -> EnvStep:
    rewards, action_mask, player, valid, obs, state_tensor = zip(*[self._prepare_env_step(state) for state in states]) 

    return EnvStep(
        obs=np.asarray(obs, dtype=np.float32),
        state = np.asarray(state_tensor, dtype=np.float32),
        legal=np.asarray(action_mask, dtype=np.int8),
        player_id=np.asarray(player, dtype=np.int8),
        valid=np.asarray(valid, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32))
  

  def _batch_of_states_apply_action(
      self, states: Sequence[pyspiel.State],
      actions: chex.Array) -> Sequence[pyspiel.State]:
    """Apply a batch of `actions` to a parallel list of `states`."""
    for state, action in zip(states, list(actions)):
      if not state.is_terminal():
        if action not in state.legal_actions():
          return False, states
        self.actor_steps += 1
        state.apply_action(action)
        self._play_chance(state)
    return True, states

  def _play_chance(self, state: pyspiel.State) -> pyspiel.State:
    """Plays the chance nodes until we end up at another type of node.

    Args:
      state: to be updated until it does not correspond to a chance node.
    Returns:
      The same input state object, but updated. The state is returned
      only for convenience, to allow chaining function calls.
    """
    while state.is_chance_node():
      chance_outcome, chance_proba = zip(*state.chance_outcomes())
      action = self._np_rng.choice(chance_outcome, p=chance_proba)
      state.apply_action(action)
    return state
