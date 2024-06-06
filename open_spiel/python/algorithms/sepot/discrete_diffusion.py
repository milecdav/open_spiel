import jax
import jax.numpy as jnp
import chex
import numpy as np
import pyspiel
import flax.linen as nn
from flax.training import train_state
import optax
import functools
import ast
import pickle

from open_spiel.python.algorithms import get_all_states
from open_spiel.python.algorithms.sepot.diffusion import ConditionalDiffusionModel, SinusoidalEmbedding, TimeEmbedding, DiffusionModel, conditional_diffuse_helper, diffuse_helper
from open_spiel.python.policy import TabularPolicy


# TODO this works only for 2 classes
def conditional_discrete_diffuse_helper(module, x, c, t):
  return conditional_diffuse_helper(module, x, c, t).reshape(*x.shape[:-1], -1, 2)

def discrete_diffuse_helper(module, x, t):
  return diffuse_helper(module, x, t).reshape(*x.shape[:-1], -1, 2)

class ConditionalDiscreteDiffusionModel(nn.Module):
  in_dim: int
  cond_dim:int
  hidden_dims: int
  @nn.compact
  def __call__(self, x, c, t):
    # Sinusoidal Embedding
    orig_shape = x.shape
    x = x.reshape(*x.shape[:-2], -1) # Flattening the tensor
    sin_embedding = SinusoidalEmbedding(self.in_dim)(t)
    
    # Embedding for the conditional task
    conditional_embedding = nn.Dense(self.cond_dim)(c)
    conditional_embedding = nn.silu(conditional_embedding)
    conditional_embedding = nn.Dense(self.cond_dim)(conditional_embedding)
    
    # Concatenating the embeddings and creating the time embedding out of them
    sin_embedding = jnp.concatenate([sin_embedding, conditional_embedding], axis=-1)
    time_embedding = TimeEmbedding(self.in_dim * 4, self.in_dim * 4)(sin_embedding)
    
    time_embedding_1 = nn.Dense(self.in_dim)(time_embedding)
    x = jnp.concatenate([x, time_embedding_1], axis=-1)
    x = nn.Dense(self.hidden_dims)(x)
    x = nn.silu(x)
    
    time_embedding_2 = nn.Dense(self.hidden_dims)(time_embedding)
    x = jnp.concatenate([x, time_embedding_2], axis=-1)
    x = nn.Dense(self.hidden_dims)(x)
    x = nn.silu(x)
    x = nn.Dense(self.in_dim)(x) # Returning the same dimension as input
    
    return x.reshape(orig_shape)
  
  def diffuse(self, x, c, t):
    return self(x, c, t)
   

class DiscreteDiffusion():
  def __init__(self, game_name, game_params, noise_steps=500, conditional=True, encoder_decoder=False, cond_dim=128, latent_dim=256, hidden_dim=256, sampled_trajectories=10, training_regime="all", clamp_result=True, seed=42, stop_training_encoder=9999999999):
    self.game_name = game_name
    self.game_params = game_params
    self.game = pyspiel.load_game_as_turn_based(game_name, game_params)
    self.game_state = self.game.new_initial_state()
    self.noise_steps = noise_steps
    self.conditional = conditional
    self.q_matrix, self.q_one_step_matrix = self.compute_matrices()
    # print(self.q_matrix.shape)
    self.initial_seed = seed
    self.rng_key = jax.random.PRNGKey(seed)
    self.np_rng_key = np.random.default_rng(seed)
    self.eps = 1e-6
    self.classes =2
    self.training_regime = training_regime
    self.sampled_trajectories = sampled_trajectories
    
    self.cond_dim = cond_dim
    self.latent_dim = latent_dim
    self.hidden_dim = hidden_dim
    
    self.temp_state_tensor = jnp.asarray([self.game_state.state_tensor()])
    self.temp_state_tensor_logits = jnp.log(jax.nn.one_hot(self.temp_state_tensor, self.classes) + self.eps)
    self.temp_public_state_tensor = jnp.asarray([self.game_state.public_state_tensor()])
    
    if conditional and not encoder_decoder:
      self.model = ConditionalDiscreteDiffusionModel(self.temp_state_tensor.shape[-1] * self.classes,  cond_dim, hidden_dim)
    elif not conditional and not encoder_decoder:
      # self.model = DiffusionModel(self.game.state_tensor_shape()[0], hidden_dim)  
      self.model = DiffusionModel(self.temp_state_tensor.shape[-1]  * self.classes, hidden_dim)
    else:
      assert False, "Not implemented encoder_decoder"
      
      
      
    self.policy = None
    # self.policy.action_probability_array
    # Policy that is random
    if self.training_regime == "traj_p":
      self.policy = TabularPolicy(self.game)
      temp_policy = self.np_rng_key.uniform(low=0, high=1, size=(self.policy.action_probability_array.shape))
      temp_policy *= self.policy.legal_actions_mask
      temp_policy /= np.sum(temp_policy, axis=1, keepdims=True)
      self.policy.action_probability_array = temp_policy
    
    # self.params = self.model.init_params(self.next_key())
    self.params = self.model.init(self.next_key(), self.temp_state_tensor_logits, self.temp_public_state_tensor, jnp.asarray([[0]])) if self.conditional else self.model.init(self.next_key(), self.temp_state_tensor, jnp.asarray([[0]]))
    
    self.cache_states, self.cache_conditions = None, None 
    if conditional:
      self.diff_method = nn.apply(conditional_diffuse_helper, self.model)
    else:
      self.diff_method = nn.apply(diffuse_helper, self.model)
      
    self.optimizer = optax.chain(
        # optax.clip_by_global_norm(1.0),
        optax.adam(3e-4)
    )
    # print("train_state")
    self.train_state = train_state.TrainState.create(apply_fn=self.model.apply, params=self.params, tx=self.optimizer)  
      
      
      
  def __getstate__(self):
    policy = []
    if self.policy is not None:
      policy = self.policy.action_probability_array
    """To serialize the agent."""
    return dict(
        game_name=self.game_name,
        game_params=self.game_params,
        noise_steps=self.noise_steps,
        cond_dim=self.cond_dim,
        latent_dim=self.latent_dim,
        hidden_dim=self.hidden_dim,
        conditional=self.conditional,
        training_regime=self.training_regime,
        # clamp_result = self.clamp_result,
        sampled_trajectories = self.sampled_trajectories,
        # encoder_decoder = self.encoder_decoder,
        params=self.train_state.params,
        rng_key = self.rng_key,
        np_rng_key = self.np_rng_key,
        initial_seed = self.initial_seed,
        policy=policy
        # policy = self.policy.action_probability_array,
    )
    
  def __setstate__(self, state):
    self.__init__(
      game_name=state["game_name"],
      game_params=state["game_params"],
      noise_steps=state["noise_steps"],
      cond_dim=state["cond_dim"],
      latent_dim=state["latent_dim"],
      hidden_dim=state["hidden_dim"],
      conditional=state["conditional"], 
      training_regime=state["training_regime"],
      # clamp_result=state["clamp_result"],
      sampled_trajectories=state["sampled_trajectories"],
      # encoder_decoder=state["encoder_decoder"],
      seed = state["initial_seed"]
    )
    self.params = state["params"]
    self.rng_key = state["rng_key"]
    self.np_rng_key = state["np_rng_key"]
    if len(state["policy"]) >0:
      self.policy = TabularPolicy(self.game)
      self.policy.action_probability_array = state["policy"]
    self.train_state = train_state.TrainState.create(apply_fn=self.model.apply, params=self.params, tx=optax.adam(3e-4))  # Is this valid?
    
  def next_key(self):
    self.rng_key, key = jax.random.split(self.rng_key)
    return key
  def next_keys(self, n=2):
    self.rng_key, *keys = jax.random.split(self.rng_key, n+1)
    return keys
  def transform_from_positive(self, states):
    return 2 * states - 1

  def transform_to_positive(self, states):
    return (states + 1) / 2
  
  
  # bot x_0 and x_t are class id
  def q_posterior_logits(self, q_one_step, q_mat, x_0, x_t, t):
    fact1 = jnp.transpose(q_one_step, (0, 2, 1))[t, x_t]
    t_1 = jnp.where(t == 0, t, t-1) # Why?
    fact2 = q_mat[t_1, x_0]
    logits = jax.nn.one_hot(x_0, self.classes) 
    out = jnp.log(fact1 + self.eps) - jnp.log(fact2 + self.eps)
    out = jnp.where(t[..., jnp.newaxis] == 0, logits, out)
    return out
  
  def q_posterior_logits_with_logits(self, q_one_step, q_mat, x_0_logits, x_t, t):
    fact1 = jnp.transpose(q_one_step, (0, 2, 1))[t, x_t]
    t_1 = jnp.where(t == 0, t, t-1) # Why?
    # fact2 = self.q_matrix[t_1, x_0]
    x_0 = jax.nn.softmax(x_0_logits, axis=-1)
    fact2 = jnp.einsum("b...c,bcd->b...d", x_0, q_mat[t_1.squeeze()])
    # logits = jax.nn.one_hot(x_0, self.classes) 
    out = jnp.log(fact1 + self.eps) - jnp.log(fact2 + self.eps)
    out = jnp.where(t[..., jnp.newaxis] == 0, x_0_logits, out)
    return out
  
  def q_sample(self, q_mat, x_0, t, noise):
    q_logits = jnp.log(q_mat[t, x_0] + self.eps) # Shape (Batch, Tensor_size, Classes)
    clipped_noise = jnp.clip(noise, self.eps, 1.0 - self.eps)
    gumbel_noise = -jnp.log(-jnp.log(clipped_noise))
    states_t = jnp.argmax(q_logits + gumbel_noise, axis=-1) #x_t
    return states_t
    
  def p_logits(self, q_one_step, q_mat, model_logits, x, t):
    out_logits = jnp.where(t[..., jnp.newaxis] == 0, model_logits, self.q_posterior_logits_with_logits(q_one_step, q_mat, model_logits, x, t))
    return out_logits, model_logits
  
  
  
  @functools.partial(jax.jit, static_argnums=(0, ))
  def training_step(self, train_state, states_0, conditions, q_one_step, q_mat, timestep_key, eps_key):
    conditions = self.transform_from_positive(conditions)
    states_0 = states_0.astype(jnp.int32)
    states_0_logits = jnp.log(jax.nn.one_hot(states_0, self.classes) + self.eps) # Shape (Batch, Tensor_size, Classes)
    
    timesteps = jax.random.choice(timestep_key, self.noise_steps, shape=(states_0.shape[0], 1))
    timesteps_normalized = timesteps/self.noise_steps
    noise = jax.random.uniform(eps_key, shape=(*states_0.shape, self.classes)) # Shape (Batch, Tensor_size, Classes)
    
    #q_sample
    q_logits = jnp.log(q_mat[timesteps, states_0] + self.eps) # Shape (Batch, Tensor_size, Classes)
    clipped_noise = jnp.clip(noise, self.eps, 1.0 - self.eps)
    gumbel_noise = -jnp.log(-jnp.log(clipped_noise))
    states_t = jnp.argmax(q_logits + gumbel_noise, axis=-1) #x_t
    states_t_logits = jnp.log(jax.nn.one_hot(states_t, self.classes) + self.eps)
    
    
    def loss_fn(params):
      true_logits = self.q_posterior_logits(q_one_step, q_mat, states_0, states_t, timesteps)
      model_output = self.diff_method(params, states_t_logits, conditions, timesteps_normalized)
      model_logits, pred_x_start_logits = self.p_logits(q_one_step, q_mat, model_output, states_0, timesteps)
      kl = jax.lax.stop_gradient(jax.nn.softmax(true_logits + self.eps, axis=-1)) * (jax.lax.stop_gradient(jax.nn.log_softmax(true_logits + self.eps, axis=-1)) - jax.nn.log_softmax(model_logits + self.eps, axis = -1))
      cat_log_lokelihood = jax.nn.log_softmax(pred_x_start_logits, axis=-1)  * states_0_logits
      vb_loss = jnp.where(timesteps[..., jnp.newaxis] == 0, cat_log_lokelihood, kl)
      vb_loss = jnp.mean(vb_loss)
      
      ce_loss = optax.softmax_cross_entropy_with_integer_labels(pred_x_start_logits, jax.lax.stop_gradient(states_0))
      # I think vb_loss is broken
      return 0.01 * vb_loss + jnp.mean(ce_loss)
      # return jnp.mean(ce_loss)
  
    
    # x_t
    # def loss_fn(params):
    #   #vb_terms_bpd state_0, states_t, t
    #   #q_posterior_logits False
    #   t_minus_one_prediction_from_t = jnp.transpose(self.q_one_step_matrix, (0, 2, 1))[timesteps, states_t] # fact1
    #   t_minus_one_prediction_from_0 = q_mat[timesteps, states_0]
    #   t_true_logits = jnp.log(t_minus_one_prediction_from_t + self.eps) + jnp.log(t_minus_one_prediction_from_0 + self.eps)
    #   # p_logits
    #   predicted_xt_logits = self.diff_method(params, t_true_logits, conditions, timesteps_normalized)
    #   # q_posterior_logits predicted_xt_logits, states_t, t, True
    #   q_posterior_fact_1 = jnp.transpose(self.q_one_step_matrix, (0, 2, 1))[timesteps, states_t]
    #   q_posterior_policy = jax.nn.softmax(predicted_xt_logits, axis=-1)
    #   q_posterior_fact_2 = jnp.einsum("b...c,bcd->b...d", q_posterior_policy, q_mat[timesteps.squeeze() - 1])
    #   q_posterior_out = jnp.log(q_posterior_fact_1 + self.eps) - jnp.log(q_posterior_fact_2 + self.eps)
    #   # model_logits
    #   q_posterior_out = jnp.where(timesteps[..., jnp.newaxis] == 0, predicted_xt_logits, q_posterior_out)
      
    #   kl = jax.lax.stop_gradient(jax.nn.softmax(t_true_logits + self.eps, axis=-1)) * (jax.lax.stop_gradient(jax.nn.log_softmax(t_true_logits + self.eps, axis=-1)) - jax.nn.log_softmax(q_posterior_out + self.eps, axis = -1))
    #   cat_log_lokelihood = jax.nn.log_softmax(predicted_xt_logits, axis=-1)  * states_0_logits
    #   vb_loss = jnp.where(timesteps[..., jnp.newaxis] == 0, cat_log_lokelihood, kl)
    #   vb_loss = jnp.mean(vb_loss)
      
    #   ce_loss = optax.softmax_cross_entropy_with_integer_labels(predicted_xt_logits, jax.lax.stop_gradient(states_0))
      
    #   return  vb_loss
  
    grad_fn = jax.value_and_grad(loss_fn, has_aux = False)
    loss, grad = grad_fn(train_state.params)
    train_state = train_state.apply_gradients(grads=grad)
    # print("finished")
    return train_state, loss
  
  @functools.partial(jax.jit, static_argnums=(0,))
  def training_step2(self, train_state, states_0, conditions, q_mat, timestep_key, eps_key):
    conditions = self.transform_from_positive(conditions)
    states_0 = states_0.astype(jnp.int32)
    # t =
    states_0_logits = jnp.log(jax.nn.one_hot(states_0, self.classes) + self.eps) # Shape (Batch, Tensor_size, Classes)
    timesteps = jax.random.choice(timestep_key, self.noise_steps, shape=(states_0.shape[0], 1))
    timesteps_normalized = timesteps/self.noise_steps
    noise = jax.random.uniform(eps_key, shape=(*states_0.shape, self.classes)) # Shape (Batch, Tensor_size, Classes)
    logits = jnp.log(q_mat[timesteps, states_0] + self.eps) # Shape (Batch, Tensor_size, Classes)
    
    
    clipped_noise = jnp.clip(noise, self.eps, 1.0 - self.eps)
    gumbel_noise = -jnp.log(-jnp.log(clipped_noise))
    states_t = jnp.argmax(logits + gumbel_noise, axis=-1) #x_t
    
    t_minus_one_prediction_from_t = jnp.transpose(q_mat, (0, 2, 1))[timesteps, states_t] # Shape (Batch, Tensor_size, Classes)
    
    
    # for b in range(states_0.shape[0]):
    #   for s_index in range(states_0.shape[1]):
    #     for c in range(self.classes):
    #       assert t_minus_one_prediction_from_t[b, s_index, c] == q_mat[timesteps[b], c, states_t[b, s_index]]
    # print("assert ok")
    states_0_softmaxed = jax.nn.softmax(states_0_logits, axis=-1)
    t_minus_one_prediction_from_0 = jnp.einsum("b...c,bcd->b...d",states_0_softmaxed, q_mat[timesteps.squeeze() - 1])
    # t_minus_one_prediction_from_0 = jnp.sum(states_0_softmaxed * t_minus_one_prediction_from_t, axis=-1)
    
    
    true_q_posterior_logits = jnp.log(t_minus_one_prediction_from_t + self.eps) - jnp.log(t_minus_one_prediction_from_0 + self.eps)
    
    true_q_posterior_logits = jnp.where(timesteps[..., jnp.newaxis] == 0, states_0_logits, true_q_posterior_logits)
    
    def loss_fn(params):
      predicted_x0_logits = self.diff_method(params, states_0_logits, conditions, timesteps_normalized)
      
      # q_posterior_logits
      
      t_minus_one_prediction_from_predicted_0 = jnp.transpose(q_mat, (0, 2, 1))[timesteps, states_0] # Shape (Batch, Tensor_size, Classes)
      # time_zero_logits = jnp.log()
      
      # predicted_x0_probabilities = jax.nn.softmax(predicted_x0_logits, axis=-1)
      # t_minus_one_prediction_from_predicted_0 = jnp.einsum("b...c,bcd->b...d",predicted_x0_probabilities, q_mat[timesteps.squeeze() - 1])
      # This is just 
      predicted_q_posterior_logits = jnp.log(t_minus_one_prediction_from_t + self.eps) - jnp.log(t_minus_one_prediction_from_predicted_0 + self.eps)
    
      predicted_q_posterior_logits = jnp.where(timesteps[..., jnp.newaxis] == 0, states_0_logits, predicted_q_posterior_logits)
      vb_loss = jax.lax.stop_gradient(jax.nn.softmax(true_q_posterior_logits, axis=-1)) * (jax.lax.stop_gradient(jax.nn.log_softmax(true_q_posterior_logits)) - jax.nn.log_softmax(predicted_q_posterior_logits))
      vb_loss = jnp.mean(jnp.sum(vb_loss, -1))
      
      
      
      ce_loss = optax.softmax_cross_entropy_with_integer_labels(predicted_x0_logits, jax.lax.stop_gradient(states_0))
      # return vb_loss
      return jnp.mean(ce_loss) + 0.01 * vb_loss
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux = False)
    loss, grad = grad_fn(train_state.params)
    train_state = train_state.apply_gradients(grads=grad)
    # print("finished")
    return train_state, loss
  
  # Shapes, states_0 = (Batch, Tensor_size), q_mat (Timesteps, Classes, Classes)
  def training(self, iterations = 1000):
    for i in range(iterations):
      time_step_key, eps_key = self.next_keys(2)
        
      if self.training_regime == "all":
        if self.cache_states is None:
          self.cache_states, self.cache_conditions = self.generate_all_states()
          # self.cache_states = jnp.concatenate([self.cache_states, np.zeros((self.cache_states.shape[0], self.latent_dim - self.cache_states.shape[1]))], axis=-1)
        states, conditions = self.cache_states, self.cache_conditions
        # self.np_rng_key.shuffle(states)
        # self.np_rng_key.shuffle(conditions)
      elif self.training_regime == "traj":
        states, conditions = self.generate_trajectories_without_policy()
      elif self.training_regime == "traj_p":
        states, conditions = self.generate_trajectories_with_policy()
        
      if self.conditional:
        self.train_state, loss = self.training_step(self.train_state, states, conditions, self.q_one_step_matrix, self.q_matrix, time_step_key, eps_key)
      else:
        assert False, "Not implemented"
  
  def get_sample_timesteps(self, batch_size, timestep):
    return jnp.full((batch_size, 1), timestep)
  
  @functools.partial(jax.jit, static_argnums=(0,))
  def ddpm_conditional_single_sample_step(self, train_state, noised_sample, conditions, z, timestep, q_mat):
    timesteps = self.get_sample_timesteps(noised_sample.shape[0], timestep)
    timesteps_normalized = timesteps/self.noise_steps
    noised_sample_logits = jnp.log(jax.nn.one_hot(noised_sample, self.classes) + self.eps) # Shape (Batch, Tensor_size, Classes)
    predicted_x0_logits = self.diff_method(train_state.params, noised_sample_logits, conditions, timesteps_normalized)
    t_minus_one_prediction_from_t = jnp.transpose(q_mat, (0, 2, 1))[timesteps, noised_sample] # Shape (Batch, Tensor_size, Classes)
    
    states_0_softmaxed = jax.nn.softmax(predicted_x0_logits, axis=-1)
    t_minus_one_prediction_from_0 = jnp.einsum("b...c,bcd->b...d",states_0_softmaxed, q_mat[timesteps.squeeze() - 1])
     
    pred_q_posterior_logits = jnp.log(t_minus_one_prediction_from_t + self.eps) - jnp.log(t_minus_one_prediction_from_0 + self.eps)
    
    pred_q_posterior_logits = jnp.where(timesteps[..., jnp.newaxis] == 0, predicted_x0_logits, pred_q_posterior_logits)
    noise = jnp.clip(z, self.eps, 1.0 - self.eps)
    
    gumbel_noise = -jnp.log(-jnp.log(noise))
    gumbel_noise = jnp.where(timesteps[..., jnp.newaxis] == 0, 0.0, gumbel_noise)
    sample = jnp.argmax(pred_q_posterior_logits + gumbel_noise, axis=-1)
    return sample
  
  def sample(self, condition_state=None, amount_samples = 10, algorithm="ddpm"):
    state_dim = self.game.state_tensor_shape()[0]
    # state_dim = self.latent_dim
    # state_dim = self.game.state_tensor_shape()[0]
    # noised_sample = jax.random.normal(self.next_key(), shape = (amount_samples, state_dim, self.classes))
    noised_sample = jax.random.randint(self.next_key(), shape = (amount_samples, state_dim), minval=0, maxval=self.classes)
    # noised_sample = jax.nn.one_hot(noised_sample, self.classes)
    # noised_sample = jnp.log(noised_sample + self.eps)
    if self.conditional:
      condition = jnp.asarray([condition_state.public_state_tensor() for _ in range(amount_samples)])
      condition = self.transform_from_positive(condition)
    for t in range(self.noise_steps - 1, -1, -1):
      if t > 0:
        z = jax.random.normal(self.next_key(), shape = (amount_samples, state_dim, self.classes))
      else:
        z = jnp.zeros((amount_samples, state_dim, self.classes))
      if algorithm == "ddpm":
        if self.conditional: 
          noised_sample = self.ddpm_conditional_single_sample_step(self.train_state, noised_sample, condition, z, t, self.q_matrix)
        else:
          noised_sample = self.ddpm_single_sample_step(self.train_state, noised_sample, z, t)
      else:
        assert False, "Not implemented"
      # if self.clamp_result:
      #   noised_sample = jax.lax.clamp(-1.0, noised_sample, 1.0) # clamping?
    # if self.encoder_decoder:
      # noised_sample = self.decode_method(self.train_state.params, noised_sample)
    # transformed = self.transform_to_positive(noised_sample)
    return noised_sample
  
  
  def generate_all_states(self):
    states = get_all_states.get_all_states(
          self.game,
          depth_limit=-1,
          include_terminals=False,
          include_chance_states=False,
          stop_if_encountered=False,
          to_string=lambda s: "[" + ','.join(str(x) for x  in s.state_tensor()) + "];[" + ','.join(str(x) for x  in s.public_state_tensor()) + "]")
    
    state_tensors = [ast.literal_eval(state.split(";")[0]) for state in states]
    public_state_tensors = [ast.literal_eval(state.split(";")[1]) for state in states]
    # states = jax.random.permutation(states)
    return np.asarray(state_tensors), np.asarray(public_state_tensors)
    
  def generate_trajectories_with_policy(self):
    states = []
    public_states = []
    for _ in range(self.sampled_trajectories):
      state = self.game.new_initial_state()
      while not state.is_terminal():
        tensor = state.state_tensor()
        if tensor not in states:
          states.append(tensor)
          public_states.append(state.public_state_tensor())
        action = self.policy.sample_action(state)
        state.apply_action(action)
    return np.asarray(states), np.asarray(public_states) 
  
  
  def generate_trajectories_without_policy(self):
    states = []
    public_states = []
    for _ in range(self.sampled_trajectories):
      state = self.game.new_initial_state()
      while not state.is_terminal():
        tensor = state.state_tensor()
        if tensor not in states:
          states.append(tensor)
          public_states.append(state.public_state_tensor())
        action = self.np_rng_key.choice(state.legal_actions())
        state.apply_action(action)
    return np.asarray(states), np.asarray(public_states) 
  
  def compute_matrices(self, classes=2):
    betas = self.compute_betas(self.noise_steps)
    q_different_indices = jnp.ones((1, classes, classes)) * betas[..., jnp.newaxis, jnp.newaxis] / classes
    q_same_indices = jnp.ones((1, classes, classes)) * (1 - ((classes - 1) * betas[..., jnp.newaxis, jnp.newaxis]/classes))
    q_one_step_mats = jnp.where(jnp.eye(classes, dtype=bool)[jnp.newaxis, ...], q_same_indices, q_different_indices)
    
    @chex.dataclass(frozen=True)
    class QMatrixCarry:
      q_mat_t: jnp.ndarray
    
    def _loop_q_matrix(carry:QMatrixCarry, q_one_step_mat):
      q_mat_t = carry.q_mat_t @ q_one_step_mat
      return QMatrixCarry(q_mat_t=q_mat_t), q_mat_t
      
    _, q_mat = jax.lax.scan(f=_loop_q_matrix, init=QMatrixCarry(q_mat_t=jnp.eye(classes)), xs=q_one_step_mats)
    
    # q_mat2 = [q_one_step_mats[0]]
    # q_mat_t2 = q_one_step_mats[0]
    # for i in range(1, self.noise_steps):
    #   q_mat_t2 = q_mat_t2 @ q_one_step_mats[i]
    #   q_mat2.append(q_mat_t2)
    # q_mat2 = jnp.stack(q_mat2)
    # assert jnp.allclose(q_mat, q_mat2)
    return q_mat, q_one_step_mats
    
  def compute_betas(self, noise_steps, s = 0.008):
    steps = jnp.arange(noise_steps + 1) / noise_steps
    alpha_bar = jnp.cos((steps + s)/(1 + s) * jnp.pi / 2 )
    betas = jnp.minimum(1 - alpha_bar[1:] / alpha_bar[:-1], 0.999)
    return betas
  
  
def main():# C
  game = "goofspiel"
  game_params = {"num_cards": 4, "points_order": "descending", "imp_info": True}
  diff = DiscreteDiffusion(game, game_params)
  state2 =diff.game.new_initial_state()
  state2.apply_action(3)
  state2.apply_action(1)
  
  # batch = [diff.game.new_initial_state().state_tensor(), state2.state_tensor()]
  print(state2.state_tensor())
  # diff.training(iterations=20000)
  model_name = "test.pkl"
  # with open(model_name, "wb") as f:
    # pickle.dump(diff, f)
  # print(diff.sample(state2, 3))
  
  # with open(model_name, "rb") as f:
    # diff2 = pickle.load(f)
  # print(diff2.sample(state2, 3))
  
  
  
if __name__ == "__main__":
  main()
  
  