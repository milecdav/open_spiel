import jax
import flax.linen as nn

import numpy as np
import functools
from typing import Any, Sequence, Tuple
import chex

import jax.numpy as jnp

Params = chex.ArrayTree


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

class ResidualBlock(nn.Module):
  
  features: int
  
  @nn.compact
  def __call__(self, x, train: bool):
    residual = x
    x = nn.Conv(self.features, (3, 3))(x)
    # x = nn.BatchNorm(use_running_average=not train)(x)
    x = nn.relu(x)
    x = nn.Conv(self.features, (3, 3))(x)
    # x = nn.BatchNorm(use_running_average=not train)(x)
    x = x + residual
    x = nn.relu(x)
    return x
  
class NetworkBody(nn.Module):
  residual_blocks: Sequence[Tuple[int, int]]
  out_dims: int
  
  @nn.compact
  def __call__(self, x, train: bool):
    for blocks, block_features in self.residual_blocks:
      x = nn.Conv(block_features, (3, 3))(x)
      x = nn.relu(x)
      for _ in range(blocks):
        x = ResidualBlock(block_features)(x, train)
      x = nn.Conv(block_features, (2, 2), strides=(2, 2), padding='VALID')(x)
      x = nn.relu(x)
    x = x.reshape(*x.shape[:-3], -1) # TODO(kubicon) - What if shape is just vector o.O?
    x = nn.Dense(self.out_dims)(x)
    return x

class RNaDNetwork(nn.Module):
  """The RNaD network.""" 
  out_dims: int
  hidden_dims: int
  residual_blocks: Sequence[Tuple[int, int]]

  @nn.compact
  def __call__(self, obs: chex.Array, legal: chex.Array, train: bool = False):
    x = NetworkBody(self.residual_blocks, self.hidden_dims)(obs, train=train)
    
    logit = nn.Dense(self.out_dims)(x)       # shape inference
    v = nn.Dense(1)(x)

    pi = _legal_policy(logit, legal)
    
    return pi, v 


class RNaDSmall:
  
  def __init__(self, file_name, params, out_dims, hidden_dims, residual_blocks, np_rng, game_name, game_params):
    self.file_name = file_name
    self.params = params
    self.network = RNaDNetwork(out_dims, hidden_dims, residual_blocks)
    self.np_rng = np.random.RandomState(0)
    self.np_rng.set_state(np_rng)
    self.game_name = game_name
    self.game_params = game_params
    
    # self.game = pyspiel.load_game
    
  @functools.partial(jax.jit, static_argnums=(0, ))
  def jitted_network(self, params: Params, obs: chex.Array, legal: chex.Array) ->chex.Array: 
    pi, v = self.network.apply(params, obs, legal) 
    # pi = jnp.where(pi >= 0.02, pi, 0) # My trick
    # pi = pi / jnp.sum(pi)
    return pi
    
    
  def get_policy(self, state):
    if self.game_name == "dark_chess":
      obs = np.array(state.observation_tensor())
      obs = obs.reshape(15, 8, 8).transpose(1, 2, 0)
    elif self.game_name == "battleship":
      obs = np.array(state.information_state_tensor())
      obs = obs.reshape(state.get_game().information_state_tensor_shape()).transpose(1, 2, 0)
    legal = np.array(state.legal_actions_mask())
    pi = self.jitted_network(self.params, obs, legal)
    pi = np.array(pi, dtype=np.float64)
    pi = pi / np.sum(pi)    
    return pi
    
  def sample_action(self, state):
    pi = self.get_policy(state)
    return self.np_rng.choice(pi.shape[0], p=pi)
    
    
  def __hash__(self) -> int:
    return(hash(self.file_name))

  def __eq__(self, value: object) -> bool:
    if isinstance(value, RNaDSmall):
      return False
    return self.file_name == value.file_name


  def __getstate__(self):
    """To serialize the agent."""
    return dict(
      file_name = self.file_name,
      params = self.params,
      np_rng = self.np_rng.get_state(),
      out_dims = self.network.out_dims,
      hidden_dims = self.network.hidden_dims,
      residual_blocks = self.network.residual_blocks,
      game_name = self.game_name,
      game_params = self.game_params
    )
    
  def __setstate__(self, state):
    self.np_rng = np.random.RandomState(0)
    
    self.file_name = state["file_name"]
    self.params = state["params"]
    self.network = RNaDNetwork(state["out_dims"], state["hidden_dims"], state["residual_blocks"])
    self.np_rng.set_state(state["np_rng"])
    self.game_name = state["game_name"]
    self.game_params = state["game_params"]