import flax.linen as nn
import numpy as np
import optax
import jax.numpy as jnp
import jax
import pyspiel
from open_spiel.python.algorithms import get_all_states
import ast
from flax.training import train_state
import pickle
import random
import sys
import functools
import matplotlib.pyplot as plt

from open_spiel.python.algorithms.reconstruct_public_state import reconstruct_goofspiel, reconstruct_states_from_histories
from open_spiel.python.policy import TabularPolicy

from pyinstrument import Profiler
 
 
class SinusoidalEmbedding:
  def __init__(self, dim, theta = 10000) -> None:
    self.half_dim = dim // 2
    self.theta = theta
    
  def __call__(self, timesteps):
    embedding = jnp.log(self.theta) / (self.half_dim - 1)
    embedding = jnp.exp(jnp.arange(self.half_dim) * (-embedding))
    embedding = jnp.outer(timesteps, embedding)
    return jnp.concatenate([jnp.sin(embedding), jnp.cos(embedding)], axis=-1)
  
class TimeEmbedding(nn.Module):
  hidden_dims: int
  out_dims: int
  @nn.compact
  def __call__(self, t):
    t = nn.Dense(self.hidden_dims)(t)
    t = nn.silu(t)
    return nn.Dense(self.out_dims)(t)
    

class EncoderDecoder(nn.Module):
  in_dim: int
  hidden_dims: int 
  
  def setup(self):
    self.encode1 = nn.Dense(self.hidden_dims)
    self.encode2 = nn.Dense(self.hidden_dims)
    self.decode1 = nn.Dense(self.hidden_dims)
    self.decode2 = nn.Dense(self.in_dim)
  
  def __call__(self, x):
    encoded = self.encode(x)
    
    decoded = self.decode(encoded) 
    
    return encoded, decoded
  
  def encode(self, x):
    encoded = self.encode1(x)
    encoded = nn.silu(encoded)
    encoded = self.encode2(encoded)
    encoded = nn.tanh(encoded)
    return encoded
  
  def decode(self, x):
    decoded = self.decode1(x)
    decoded = nn.silu(decoded)
    decoded = self.decode2(decoded)
    return decoded
    
    

class DiffusionModel(nn.Module):
  in_dim: int
  hidden_dims: int
  @nn.compact
  def __call__(self, x, t):
    sin_embedding = SinusoidalEmbedding(self.in_dim)(t)
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
    
    return x
  
  def diffuse(self, x, t):
    return self(x, t)
  

class DiffusionModelEncoderDecoder(nn.Module):
  in_dim: int 
  latent_dim: int
  hidden_dims: int
  def setup(self):
    self.encoder_decoder = EncoderDecoder(self.in_dim ,self.latent_dim)
    self.diffusion_model = DiffusionModel(self.latent_dim, self.hidden_dims)
  
  # This is just so we initiate all the weights, but it should not be used!
  def __call__(self, x, t):
    encoded, decoded = self.encoder_decoder(x)
    return decoded, self.diffusion_model(encoded, t)  
  
  def encode(self, x):
    return self.encoder_decoder.encode(x)
  
  def decode(self, x):
    return self.encoder_decoder.decode(x)
  
  def encode_decode(self, x):
    return self.encoder_decoder(x)
  
  def diffuse(self, x, t):
    return self.diffusion_model(x, t)

class ConditionalDiffusionModel(nn.Module):
  in_dim: int
  cond_dim:int
  hidden_dims: int
  @nn.compact
  def __call__(self, x, c, t):
    # Sinusoidal Embedding
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
    
    return x
  
  def diffuse(self, x, c, t):
    return self(x, c, t)
   

class ConditionalDiffusionModelEncoderDecoder(nn.Module):
  in_dim: int
  cond_dim:int
  latent_dim: int
  hidden_dims: int
  def setup(self):
    self.encoder_decoder = EncoderDecoder(self.in_dim ,self.latent_dim)
    self.diffusion_model = ConditionalDiffusionModel(self.latent_dim, self.cond_dim, self.hidden_dims)
  
  # This is just so we initiate all the weights, but it should not be used!
  def __call__(self, x, c, t):
    encoded, decoded = self.encoder_decoder(x)
    return decoded, self.diffusion_model(encoded, c, t)  
  
  # def call_without_encoding(self, x, c, t):
  #   encoded, decoded = self.encoder_decoder(x)
  #   return decoded, self.diffusion_model(x, c, t)
  
  def encode(self, x):
    return self.encoder_decoder.encode(x)
  
  def decode(self, x):
    return self.encoder_decoder.decode(x)
  
  def encode_decode(self, x):
    return self.encoder_decoder(x)
  
  def diffuse(self, x, c, t):
    return self.diffusion_model(x, c, t)

def encode_decode_helper(module, x):
  return module.encode_decode(x)

def encode_helper(module, x):
  return module.encode(x)

def decode_helper(module, x):
  return module.decode(x)

def diffuse_helper(module, x, t):
  return module.diffuse(x, t)

def conditional_diffuse_helper(module, x, c, t):
  return module.diffuse(x, c, t)


class FullDiffusionModel():
  
  def __init__(self, game_name, game_params, noise_steps=500, cond_dim=128, latent_dim=256, hidden_dim=256, conditional=False, training_regime="all", clamp_result=True, sampled_trajectories=1, encoder_decoder=False, seed=42, stop_training_encoder=10000):
    self.game_name = game_name
    self.game_params = game_params
    self.game = pyspiel.load_game_as_turn_based(game_name, game_params)
    self.game_state = self.game.new_initial_state()
    self.noise_steps = noise_steps
    self.cond_dim = cond_dim
    self.latent_dim = latent_dim
    self.hidden_dim = hidden_dim
    self.conditional = conditional
    self.encoder_decoder = encoder_decoder # TODO: decoder trains well but the diffusion process breaks.
    self.clamp_result = clamp_result
    self.sampled_trajectories = sampled_trajectories
    self.initial_seed = seed
    
    self.stop_training_encoder = stop_training_encoder
    
    self.temp_state_tensor = jnp.asarray([self.game_state.state_tensor()])
    self.temp_public_state_tensor = jnp.asarray([self.game_state.public_state_tensor()])
    # self.temp_state_tensor = jnp.concatenate([self.temp_state_tensor, jnp.zeros((1, self.latent_dim - self.temp_state_tensor.shape[-1]))], axis=-1)
    
    
    if conditional and encoder_decoder:
      self.model = ConditionalDiffusionModelEncoderDecoder(self.temp_state_tensor.shape[-1], cond_dim, latent_dim, hidden_dim)
    elif not conditional and encoder_decoder:
      self.model = DiffusionModelEncoderDecoder(self.temp_state_tensor.shape[-1], latent_dim, hidden_dim)
    elif conditional and not encoder_decoder:
      self.model = ConditionalDiffusionModel(self.temp_state_tensor.shape[-1], cond_dim, hidden_dim)
    elif not conditional and not encoder_decoder:
      # self.model = DiffusionModel(self.game.state_tensor_shape()[0], hidden_dim)  
      self.model = DiffusionModel(self.temp_state_tensor.shape[-1], hidden_dim)  
  
    
    self.rng_key = jax.random.PRNGKey(seed)
    self.np_rng_key = np.random.default_rng(seed)
  
    # print("params")
    self.params = self.model.init(self.next_key(), self.temp_state_tensor, self.temp_public_state_tensor, jnp.asarray([[0.0]])) if self.conditional else self.model.init(self.next_key(), self.temp_state_tensor, jnp.asarray([[0.0]]))
    # print("aa")
    if conditional:
      self.diff_method = nn.apply(conditional_diffuse_helper, self.model)
    else:
      self.diff_method = nn.apply(diffuse_helper, self.model)
    if encoder_decoder:
      self.encode_method = nn.apply(encode_helper, self.model)
      self.decode_method = nn.apply(decode_helper, self.model)
      self.encode_decode_method = nn.apply(encode_decode_helper, self.model)
     
    self.training_regime = training_regime
    self.cache_states, self.cache_conditions = None, None
    
    self.policy = None
    # self.policy.action_probability_array
    # Policy that is random
    if self.training_regime == "traj_p":
      self.policy = TabularPolicy(self.game)
      temp_policy = self.np_rng_key.uniform(low=0, high=1, size=(self.policy.action_probability_array.shape))
      temp_policy *= self.policy.legal_actions_mask
      temp_policy /= np.sum(temp_policy, axis=1, keepdims=True)
      self.policy.action_probability_array = temp_policy
    # for infostate_key, index in self.policy.state_lookup.items():
      
    self.losses = []
      
    # timesteps = jnp.linspace(step_eps, jnp.pi/2 -step_eps, noise_steps)
    # self.alpha_bar = jnp.cos(timesteps) ** 2 # Cumproduct of alpha
    # self.alpha_bar = self.alpha_bar / self.alpha_bar[0]
    # self.alpha = self.alpha_bar[1:] / self.alpha_bar[:-1]
    # self.sqrt_prev_alpha_bar = jnp.sqrt(self.alpha_bar[:-1])
    # self.sqrt_minus_prev_alpha_bard = jnp.sqrt(1 - self.alpha_bar[:-1])
    # self.alpha_bar = self.alpha_bar[1:] # Just to have consistent dimension with alpha
    # self.sqrt_alpha_bar = jnp.sqrt(self.alpha_bar)
    # self.minus_sqrt_alpha_bar = jnp.sqrt(1 - self.alpha_bar)
    # self.beta = jnp.clip(1 - self.alpha, 0, 0.999)
    
    
    self.optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(3e-4)
    )
    # print("train_state")
    self.train_state = train_state.TrainState.create(apply_fn=self.model.apply, params=self.params, tx=self.optimizer)  
    # print("ee")
    
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
        clamp_result = self.clamp_result,
        sampled_trajectories = self.sampled_trajectories,
        encoder_decoder = self.encoder_decoder,
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
      clamp_result=state["clamp_result"],
      sampled_trajectories=state["sampled_trajectories"],
      encoder_decoder=state["encoder_decoder"],
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
  
  def transform_from_positive(self, states):
    return 2 * states - 1

  def transform_to_positive(self, states):
    return (states + 1) / 2
  
  def compute_noise_rates(self, t, step_eps=0.005):
    first_val = jnp.cos((step_eps)/ (1 + step_eps) * jnp.pi/2) ** 2
    min_val = 0.0001
    max_val = 0.9999
    
    alpha_bar = jnp.cos((t + step_eps)/ (1 + step_eps) * jnp.pi/2) ** 2
    alpha_bar = alpha_bar / first_val
    
    minus_t = jnp.maximum(t - 1/self.noise_steps, 0)
    alpha_bar_t_minus_one = jnp.cos((minus_t + step_eps) / (1 + step_eps) * jnp.pi/2) ** 2
    alpha_bar_t_minus_one = alpha_bar_t_minus_one / first_val
    
    alpha_bar = jnp.clip(alpha_bar, min_val, max_val)
    alpha_bar_t_minus_one = jnp.clip(alpha_bar_t_minus_one, min_val, max_val)
    
    alpha = alpha_bar / alpha_bar_t_minus_one
    
    beta = jnp.clip(1 - alpha, min_val, max_val)
    return alpha_bar, alpha_bar_t_minus_one, alpha, beta
  
  # Unstable 
  # def compute_noise_rates(self, t, step_eps=0.0):
  #   first_val, final_val = step_eps, jnp.pi/2 - step_eps
  #   radians_t = first_val + (final_val - first_val) * t
  #   radians_t_minus_one = first_val + (final_val - first_val) * jnp.maximum((t - 1/self.noise_steps), 0) # To prevent going ino negative radians (may not break naything, but just to be sure)

  #   alpha_bar = jnp.cos(radians_t) ** 2
  #   alpha_bar = alpha_bar / (jnp.cos(first_val) ** 2)
  #   alpha_bar_t_minus_one = jnp.cos(radians_t_minus_one) ** 2
  #   alpha_bar_t_minus_one = alpha_bar_t_minus_one / (jnp.cos(first_val) ** 2) 
  #   alpha_bar = jnp.clip(alpha_bar, min_val, max_val)
  #   alpha_bar_t_minus_one = jnp.clip(alpha_bar_t_minus_one, min_val, max_val)
  #   alpha = alpha_bar / alpha_bar_t_minus_one
    
  #   min_val = 0.0001
  #   max_val = 0.9999
  #   alpha = jnp.clip(alpha, min_val, max_val)
  #   # beta = jnp.clip(1 - alpha, 0.0001, 0.9999)
    
    
    
    
    
  #   return alpha_bar, alpha_bar_t_minus_one, alpha, beta
    
  @functools.partial(jax.jit, static_argnums=(0,))
  def training_step(self, train_state, time_step_key, eps_key, states):
    states = self.transform_from_positive(states)
    
    timesteps = jax.random.choice(time_step_key, self.noise_steps, shape=(states.shape[0], 1))
    timesteps = timesteps/self.noise_steps
    
    noise_eps, noised_states = self.generate_noise(states, timesteps, eps_key)
    
    def loss_fn(params):
      predicted_noise = self.diff_method(params, noised_states, timesteps)
      # predicted_noise = self.model.apply(params, noised_states, timesteps)
      # predicted_noise = self.diff_method(params, noised_states, timesteps)
      return jnp.mean((noise_eps - predicted_noise) ** 2)
      return jnp.mean((jax.lax.stop_gradient(noise_eps) - predicted_noise) ** 2)
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux = False)
    loss, grad = grad_fn(train_state.params)
    train_state = train_state.apply_gradients(grads=grad)
    return train_state, loss
     
  
  @functools.partial(jax.jit, static_argnums=(0,))
  def training_conditional_step(self, train_state, time_step_key, eps_key, states, conditions):
    states = self.transform_from_positive(states)
    conditions = self.transform_from_positive(conditions)
    
    timesteps = jax.random.choice(time_step_key, self.noise_steps, shape=(states.shape[0], 1))
    timesteps = timesteps/self.noise_steps
    
    noise_eps, noised_states = self.generate_noise(states, timesteps, eps_key)
    
    def loss_fn(params):
      # predicted_noise = self.model.apply(params, noised_states, conditions, timesteps)
      predicted_noise = self.diff_method(params, noised_states, conditions, timesteps)
      return jnp.mean((noise_eps - predicted_noise) ** 2)
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux = False)
    loss, grad = grad_fn(train_state.params)
    train_state = train_state.apply_gradients(grads=grad)
    return train_state, loss
  
  @functools.partial(jax.jit, static_argnums=(0,))
  def training_encoder_step(self, train_state, time_step_key, eps_key, states):
    states = self.transform_from_positive(states)
    
    timesteps = jax.random.choice(time_step_key, self.noise_steps, shape=(states.shape[0], 1))
    timesteps = timesteps/self.noise_steps
    # noise_eps = jax.random.normal(eps_key, (states.shape[0], self.latent_dim))
    # alpha_bar, _, _, _ = self.compute_noise_rates(timesteps)
    
    
    def loss_fn(params):
      encoded, decoded = self.encode_decode_method(params, states)
      noise_eps, noised_states = self.generate_noise(encoded, timesteps, eps_key)
      # noised_states = encoded * jax.lax.stop_gradient(jnp.sqrt(alpha_bar)) + jax.lax.stop_gradient(noise_eps) * jax.lax.stop_gradient(jnp.sqrt(1 - alpha_bar))
      predicted_noise = self.diff_method(params, noised_states, timesteps)
      # return jnp.mean((jax.lax.stop_gradient(noise_eps) - predicted_noise) ** 2) 
      # return jnp.mean((states - decoded) ** 2) + jnp.mean((noise_eps - predicted_noise) ** 2)
      # return jnp.mean((jax.lax.stop_gradient(noise_eps) - predicted_noise) ** 2)
      return jnp.mean((jax.lax.stop_gradient(states) - decoded) ** 2) + jnp.mean((jax.lax.stop_gradient(noise_eps) - predicted_noise) ** 2)
    
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux = False)
    loss, grad = grad_fn(train_state.params)
    train_state = train_state.apply_gradients(grads=grad)
    return train_state, loss
  
  @functools.partial(jax.jit, static_argnums=(0,))
  def training_encoder_step_no_encoder(self, train_state, time_step_key, eps_key, states):
    states = self.transform_from_positive(states)
    
    timesteps = jax.random.choice(time_step_key, self.noise_steps, shape=(states.shape[0], 1))
    timesteps = timesteps/self.noise_steps
    # noise_eps = jax.random.normal(eps_key, (states.shape[0], self.latent_dim))
    # alpha_bar, _, _, _ = self.compute_noise_rates(timesteps)
  
    
    def loss_only_diff(params):
      encoded, decoded = self.encode_decode_method(params, states)
      # encoded = jnp.concatenate([states, jnp.zeros((states.shape[0], self.latent_dim - states.shape[1]))], axis=-1)
      noise_eps, noised_states = self.generate_noise(encoded, timesteps, eps_key)
      # noised_states = encoded * jax.lax.stop_gradient(jnp.sqrt(alpha_bar)) + jax.lax.stop_gradient(noise_eps) * jax.lax.stop_gradient(jnp.sqrt(1 - alpha_bar))
      predicted_noise = self.diff_method(params, jax.lax.stop_gradient(noised_states), timesteps) 
      return jnp.mean((jax.lax.stop_gradient(noise_eps) - predicted_noise) ** 2)
    
    grad_fn = jax.value_and_grad(loss_only_diff, has_aux = False)
    loss, grad = grad_fn(train_state.params)
    train_state = train_state.apply_gradients(grads=grad)
    return train_state, loss
  
  
  
  @functools.partial(jax.jit, static_argnums=(0,))
  def training_encoder_conditional_step(self, train_state, time_step_key, eps_key, states, conditions):
    states = self.transform_from_positive(states)
    conditions = self.transform_from_positive(conditions)
    
    timesteps = jax.random.choice(time_step_key, self.noise_steps, shape=(states.shape[0], 1))
    timesteps = timesteps/self.noise_steps
    
    def loss_fn(params):
      encoded, decoded = self.encode_decode_method(params, states)
      noise_eps, noised_states = self.generate_noise(encoded, timesteps, eps_key)
      predicted_noise = self.diff_method(params, noised_states, conditions, timesteps)
      # return jnp.mean((states - decoded) ** 2) + jnp.mean((noise_eps - predicted_noise) ** 2)
      return jnp.mean((jax.lax.stop_gradient(states) - decoded) ** 2) + jnp.mean((jax.lax.stop_gradient(noise_eps) - predicted_noise) ** 2)
      
    grad_fn = jax.value_and_grad(loss_fn, has_aux = False)
    loss, grad = grad_fn(train_state.params)
    train_state = train_state.apply_gradients(grads=grad)
    return train_state, loss
    

  @functools.partial(jax.jit, static_argnums=(0,))
  def training_encoder_conditional_step_no_encoder(self, train_state, time_step_key, eps_key, states, conditions):
    states = self.transform_from_positive(states)
    conditions = self.transform_from_positive(conditions)
    
    timesteps = jax.random.choice(time_step_key, self.noise_steps, shape=(states.shape[0], 1))
    timesteps = timesteps/self.noise_steps
    
    def loss_fn(params):
      encoded, decoded = self.encode_decode_method(params, states)
      noise_eps, noised_states = self.generate_noise(encoded, timesteps, eps_key)
      predicted_noise = self.diff_method(params, jax.lax.stop_gradient(noised_states), conditions, timesteps)
      # return jnp.mean((states - decoded) ** 2) + jnp.mean((noise_eps - predicted_noise) ** 2)
      return jnp.mean((jax.lax.stop_gradient(noise_eps) - predicted_noise) ** 2)
      
    grad_fn = jax.value_and_grad(loss_fn, has_aux = False)
    loss, grad = grad_fn(train_state.params)
    train_state = train_state.apply_gradients(grads=grad)
    return train_state, loss
    
  def train_enchoder(self, train_state, anchor, positive, negative):
  
    def triplet_loss(params):
      anchor_embedding = self.encode_method(params, anchor)
      positive_embedding = self.encode_method(params, positive)
      negative_embedding = self.encode_method(params, negative)
      positive_distance = jnp.linalg.norm(anchor_embedding - positive_embedding, axis=-1)
      negative_distance = jnp.linalg.norm(anchor_embedding - negative_embedding, axis=-1)
      return jnp.mean(jnp.maximum(positive_distance - negative_distance + 1, 0))
    
    grad_fn = jax.value_and_grad(triplet_loss, has_aux = False)
    loss, grad = grad_fn(train_state.params)
    train_state = train_state.apply_gradients(grads=grad)
    return train_state, loss
    
  
  def predict_noise(self, states, timesteps):
    return self.diff_model.apply(self.train_state.params, states, timesteps)
    
  # equivalent of q-sample
  def generate_noise(self, states, timesteps, rng_key):
    noise_eps = jax.random.normal(rng_key, states.shape)
    alpha_bar, _, _, _ = self.compute_noise_rates(timesteps)
    noised_states = states * jnp.sqrt(alpha_bar) + noise_eps * jnp.sqrt(1 - alpha_bar)
    return noise_eps, noised_states

  def training(self, iterations = 1000):
    for i in range(iterations):
      # if i % 100 == 0:
      #   print(i)
      # print("e")
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
      
      if self.conditional and self.encoder_decoder:
        if i < self.stop_training_encoder:
          self.train_state, loss = self.training_encoder_conditional_step(self.train_state, time_step_key, eps_key, states, conditions)
        else:
          self.train_state, loss = self.training_encoder_conditional_step_no_encoder(self.train_state, time_step_key, eps_key, states, conditions)
      
      elif not self.conditional and self.encoder_decoder:
        if i < self.stop_training_encoder:
          self.train_state, loss = self.training_encoder_step(self.train_state, time_step_key, eps_key, states)
        else:
          self.train_state, loss = self.training_encoder_step_no_encoder(self.train_state, time_step_key, eps_key, states)
      
      elif self.conditional and not self.encoder_decoder:
        self.train_state, loss = self.training_conditional_step(self.train_state, time_step_key, eps_key, states, conditions)
        
      elif not self.conditional and not self.encoder_decoder:
        self.train_state, loss = self.training_step(self.train_state, time_step_key, eps_key, states)
      # self.losses.append(loss)
      # print(sys.getrefcount(states))
      

  def get_sample_timesteps(self, batch_size, timestep):
    return jnp.full((batch_size, 1), timestep/self.noise_steps)
  
  def compute_ddpm_prev_sample(self, predicted_noise, noised_sample, z, timesteps):
    alpha_bar, alpha_bar_prev, alpha, beta = self.compute_noise_rates(timesteps)
    # sigma_squared = beta * (1 - alpha_bar_prev)/  (1 - alpha_bar)
    return 1/(jnp.sqrt(alpha)) * (noised_sample - ( ((beta) / (jnp.sqrt(1 - alpha_bar))) * predicted_noise)) + jnp.sqrt(beta) * z
  
    # sample = jnp.sqrt(prev_alpha) * (noised_sample - jnp.sqrt(1 - alpha) * predicted_noise) / jnp.sqrt(alpha) + jnp.sqrt(1 - prev_alpha - beta) * predicted_noise + jnp.sqrt(beta) * z
  def compute_ddim_prev_sample(self, predicted_noise, noised_sample, z, timesteps):
    alpha_bar, alpha_bar_prev, _, beta = self.compute_noise_rates(timesteps)
    # return jnp.sqrt(prev_alpha) * 
    eta = 1.0
    sigma = eta * jnp.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar) * (1 - alpha_bar/alpha_bar_prev))
    # sigma = jnp.sqrt(beta)
    return jnp.sqrt(alpha_bar_prev) * (noised_sample - jnp.sqrt(1 - alpha_bar) * predicted_noise) / jnp.sqrt(alpha_bar) + jnp.sqrt(1 - alpha_bar_prev - sigma ** 2) * predicted_noise + sigma * z
  
  @functools.partial(jax.jit, static_argnums=(0,))
  def ddpm_single_sample_step(self, train_state, noised_sample, z, timestep):
    timesteps = self.get_sample_timesteps(noised_sample.shape[0], timestep)
    # predicted_noise = self.model.apply(train_state.params, noised_sample, timesteps)
    predicted_noise = self.diff_method(train_state.params, noised_sample, timesteps)
    return self.compute_ddpm_prev_sample(predicted_noise, noised_sample, z, timesteps)
    # return sample
  
  @functools.partial(jax.jit, static_argnums=(0,))
  def ddpm_conditional_single_sample_step(self, train_state, noised_sample, conditions, z, timestep):
    timesteps = self.get_sample_timesteps(noised_sample.shape[0], timestep)
    # predicted_noise = self.model.apply(train_state.params, noised_sample, conditions, timesteps)
    predicted_noise = self.diff_method(train_state.params, noised_sample, conditions, timesteps)
    return self.compute_ddpm_prev_sample(predicted_noise, noised_sample, z, timesteps)
  
  @functools.partial(jax.jit, static_argnums=(0,))
  def ddim_single_sample_step(self, train_state, noised_sample, z, timestep):
    timesteps = self.get_sample_timesteps(noised_sample.shape[0], timestep)
    # predicted_noise = self.model.apply(train_state.params, noised_sample, timesteps)
    predicted_noise = self.diff_method(train_state.params, noised_sample, timesteps)
    return self.compute_ddim_prev_sample(predicted_noise, noised_sample, z, timesteps)
  
  @functools.partial(jax.jit, static_argnums=(0,))
  def ddim_conditional_single_sample_step(self, train_state, noised_sample, conditions, z, timestep):
    timesteps = self.get_sample_timesteps(noised_sample.shape[0], timestep)
    # predicted_noise = self.model.apply(train_state.params, noised_sample, conditions, timesteps)
    predicted_noise = self.diff_method(train_state.params, noised_sample, conditions, timesteps)
    return self.compute_ddim_prev_sample(predicted_noise, noised_sample, z, timesteps)
  
  def sample(self, condition_state=None, amount_samples = 10, algorithm="ddpm"):
    state_dim = self.latent_dim if self.encoder_decoder else self.game.state_tensor_shape()[0]
    # state_dim = self.latent_dim
    # state_dim = self.game.state_tensor_shape()[0]
    noised_sample = jax.random.normal(self.next_key(), shape = (amount_samples, state_dim))
    if self.conditional:
      condition = jnp.asarray([condition_state.public_state_tensor() for _ in range(amount_samples)])
      condition = self.transform_from_positive(condition)
    for t in range(self.noise_steps - 1, -1, -1):
      if t > 0:
        z = jax.random.normal(self.next_key(), shape = (amount_samples, state_dim))
      else:
        z = jnp.zeros_like(noised_sample)
      if algorithm == "ddpm":
        if self.conditional: 
          noised_sample = self.ddpm_conditional_single_sample_step(self.train_state, noised_sample, condition, z, t)
        else:
          noised_sample = self.ddpm_single_sample_step(self.train_state, noised_sample, z, t)
      elif algorithm == "ddim":
        if self.conditional:
          noised_sample = self.ddim_conditional_single_sample_step(self.train_state, noised_sample, condition, z, t)
        else:
          noised_sample = self.ddim_single_sample_step(self.train_state, noised_sample, z, t)
      if self.clamp_result:
        noised_sample = jax.lax.clamp(-1.0, noised_sample, 1.0) # clamping?
    if self.encoder_decoder:
      noised_sample = self.decode_method(self.train_state.params, noised_sample)
    transformed = self.transform_to_positive(noised_sample)
    return transformed
  
 
def test_runnability():
  for c in [False, True]:
    for ec in [False, True]:
      print("I")
      model = FullDiffusionModel("goofspiel", {"num_cards": 3, "imp_info": True, "points_order": "descending"}, conditional=c, encoder_decoder=ec)
      model.training(10)
      state = model.game.new_initial_state()
      rng_key = jax.random.PRNGKey(654)
      for alg in ["ddpm", "ddim"]:
        model.rng_key = rng_key
        with open("test.pkl", "wb") as f:
          pickle.dump(model, f)
        samples = model.sample(state, 10, alg)
        
        with open("test.pkl", "rb") as f:
          model2 = pickle.load(f)
        model2.rng_key = rng_key
        samples2 = model2.sample(state, 10, alg)
        assert jnp.allclose(samples, samples2)
  
def train_to_eval():
  model_name = "test2.pkl"
  model = FullDiffusionModel("goofspiel", {"num_cards": 4, "imp_info": True, "points_order": "descending"}, training_regime="all", sampled_trajectories=20, clamp_result=False, conditional=False, encoder_decoder=False, latent_dim = 128)
  # state = model.game.new_initial_state()
  # samples = model.sample(state, 2, "ddpm")
  # for i in samples:
  #   print(i)
  model.training(3000)
  with open(model_name, "wb") as f:
    pickle.dump(model, f)
  state = model.game.new_initial_state()
  samples = model.sample(state, 2, "ddpm")
  
  # x = np.arange(len(model.losses))

  # plt.plot(x, model.losses)
  # plt.show()

  
  for i in samples:
   print(i)
  with open(model_name, "rb") as f:
    model2 = pickle.load(f)
  state = model2.game.new_initial_state()
  samples = model2.sample(state, 2, "ddpm")
  print(jnp.asarray(state.state_tensor()))
  # state_tensor_non = model2.transform_from_positive(jnp.asarray(state.state_tensor()))
  # predicted = model2.encode_decode_method(model2.params, state_tensor_non)[1]
  # print(model2.transform_to_positive(predicted))
  for i in samples:
    print(i)
  
def train_diff(model_name): 
  model = FullDiffusionModel("goofspiel", {"num_cards": 4, "imp_info": True, "points_order": "descending"}, conditional = False,  encoder_decoder=True, training_regime="all", sampled_trajectories=20, clamp_result=True)
  profiler = Profiler()
  profiler.start()
  # print(model.params)
  model.training(20000)
  print(model.train_state.step)
  profiler.stop()
  print(profiler.output_text(color=True, unicode=True))
  
  with open(model_name, "wb") as f:
    pickle.dump(model, f)

# def eval_existency(model_name):
#   with open(model_name, "rb") as f:
#     model = pickle.load(f)
#   state = model.game.new_initial_state()
#   samples = model.sample(state, 10, "ddpm")
#   for i in samples:
#     print(i)


def eval_d(model_name):
  with open(model_name, "rb") as f:
    model = pickle.load(f)
  if model.clamp_result is False:
    model.clamp_result = True
  state = model.game.new_initial_state()
  state.apply_action(0)
  state.apply_action(3)
  # print(model.encode_method(model.params, jnp.asarray([state.state_tensor()])))
  # tensor = jnp.asarray([state.state_tensor()])
  # positive_tensor = model.transform_from_positive(tensor)
  # enc, dec = model.encode_decode_method(model.params, positive_tensor)
  # print(enc)
  # print(jnp.asarray([state.state_tensor()]))
  # print(model.transform_to_positive(dec))
  samples = model.sample(model.game.new_initial_state(), 10, "ddpm")
  for i in samples:
    print(i)

def eval_diff(model_name):
  
  with open(model_name, "rb") as f:
    model = pickle.load(f)
  
  conditional_state = model.game.new_initial_state()
  conditional_state.apply_action(0)
  conditional_state.apply_action(3)
  # conditional_state.apply_action(4)
  # conditional_state.apply_action(2)
  all_histories, _ = reconstruct_goofspiel(conditional_state, 100000)
  state_tensors = []
  probs = []
  for h in all_histories:
    p = 1.0
    state = model.game.new_initial_state()
    for i, action in enumerate(h):
      p *= model.policy.action_probabilities(state)[action]
      state.apply_action(action)
    probs.append(p)
    state_tensors.append(state.state_tensor())
  probs = np.asarray(probs)
  print(np.sum(probs))
  probs /= np.sum(probs)
  state_tensors = np.asarray(state_tensors)
  samples = model.sample(conditional_state, 10000, "ddim")
  was_closest = np.zeros(state_tensors.shape[0])
  complete_closest = np.zeros(state_tensors.shape[0])
  for sample in samples:
    # print(sample)
    s = (sample > 0.5).astype(jnp.float32)
    closest_dist = 1000000
    closest_index = -1
    for i_st, st in enumerate(state_tensors):
      dist = jnp.linalg.norm(s - st)
      if dist < closest_dist:
        closest_dist = dist
        closest_index = i_st
    was_closest[closest_index] += 1
    complete_closest[closest_index] += closest_dist
  print(was_closest)
  print(complete_closest/jnp.maximum(was_closest, 1))
  print(was_closest / np.sum(was_closest))
  print(probs)
      
def test_ed():
  game = pyspiel.load_game_as_turn_based("goofspiel", {"num_cards": 4, "imp_info": True, "points_order": "descending"})
  state = game.new_initial_state()
  model = DiffusionModelEncoderDecoder(game.state_tensor_shape()[0], 128, 256, 256)
  rng_key = jax.random.PRNGKey(654)
  params = model.init(rng_key, jnp.asarray([state.state_tensor()]), jnp.asarray([state.public_state_tensor()]), jnp.asarray([[0]]))
  ec_fn = nn.apply(encode_decode_helper, model)
  diff_fn = nn.apply(diffuse_helper, model)
  e, d = ec_fn(params, jnp.asarray([state.state_tensor()]))
  pred_noise = diff_fn(params, e, jnp.asarray([state.public_state_tensor()]), jnp.asarray([[0]]))
  print(e)
  print(d)
  print(pred_noise)
  
  
def test_noise():
  model = FullDiffusionModel("goofspiel", {"num_cards": 5, "imp_info": True, "points_order": "descending"}, conditional = True, training_regime="traj", sampled_trajectories=10, clamp_result=False, noise_steps=500)
  timesteps = jnp.linspace(0, 1, model.noise_steps)
  alpha_bar, alpha_bar_prev, alpha, beta = model.compute_noise_rates(timesteps)
  alpha2 = 1-beta
  alpha_bar2 = jnp.cumprod(alpha2)
  alpha_bar_prev2 = jnp.concatenate([jnp.asarray([1]), jnp.cumprod(alpha2[:-1])])
  print(jnp.allclose(alpha_bar, alpha_bar2))
  
  
if __name__ == "__main__":
  jnp.set_printoptions(precision=3, threshold=sys.maxsize)
  # model_name = "diffusion_models/goofspiel_5_descending/model_ns500_c1_ed0_cd128_ld256_hd256_s42_t500000.pkl"
  # test_noise()
  # eval_diff(model_name)
  model_name = "diffusion_models/goofspiel_5_descending/model_ns500_c1_ed1_cd128_ld128_hd256_s42_t1000000.pkl"
  train_to_eval()
  # train_diff(model_name)
  # eval_d(model_name)
  # eval_d(model_name)
  
  # ddpm_train_loop()
  # ddpm_train_loop()
# Forward diffusion process: q(x_t | x_0)
# def q_sample(x0, t,False noise):
#   sqrt_alpha_cumprod_t = jnp.sqrt(alphas_cumprod[t])
#   sqrt_one_minus_alpha_cumprod_t = jnp.sqrt(1 - alphas_cumprod[t])
#   return sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise

# # Reverse process: p(x_{t-1} | x_t)


# def p_sample(model, x_t, t, noise):
#   t = jnp.broadcast_to(t, x_t.shape)
#   eps_theta = model(x_t, t)
#   alpha_t = alphas[t]
#   alpha_prev_t = alphas_prev[t]
#   beta_t = betas[t]
#   pred_noise = (x_t - jnp.sqrt(alpha_t) * eps_theta) / jnp.sqrt(1 - alpha_t)
#   x_t_minus_1 = jnp.sqrt(alpha_prev_t) * pred_noise + \
#       jnp.sqrt(1 - alpha_prev_t) * noise
#   return x_t_minus_1

# # Model definition (a simple MLP)


# def model(x, t):
#   t_embedding = jnp.sin(t[:, None] * jnp.arange(10, dtype=jnp.float32))
#   x_t = jnp.concatenate([x, t_embedding], axis=-1)
#   x_t = jax.nn.relu(jax.nn.Dense(256)(x_t))
#   x_t = jax.nn.relu(jax.nn.Dense(256)(x_t))
#   return jax.nn.Dense(1)(x_t)

# # Mean squared error loss


# def loss_fn(params, x0, t, noise):
#   x_t = q_sample(x0, t, noise)
#   eps_theta = model.apply(params, x_t, t)
#   return jnp.mean((noise - eps_theta) ** 2)


# # Define hyperparameters
# timesteps = 1000
# alphas = jnp.linspace(0.0001, 0.02, timesteps)
# betas = 1 - alphas
# alphas_cumprod = jnp.cumprod(alphas)
# alphas_prev = jnp.roll(alphas, shift=1)
# alphas_prev = jax.ops.index_update(alphas_prev, 0, 1.0)

# # Initialize model and optimizer
# rng = jax.random.PRNGKey(0)
# params = model.init(rng, jnp.ones([1, 1]), jnp.ones([1, 1]))
# optimizer = optax.adam(learning_rate=1e-3)
# opt_state = optimizer.init(params)

# # Training loop


# @jax.jit
# def train_step(params, opt_state, x0, t, noise):
#   grads = jax.grad(loss_fn)(params, x0, t, noise)
#   updates, opt_state = optimizer.update(grads, opt_state)
#   params = optax.apply_updates(params, updates)
#   return params, opt_state


# # Dummy data
# x0 = jnp.array([[0.5], [-0.5]])
# timesteps = jnp.array([100, 200])
# noise = jax.random.normal(rng, shape=x0.shape)

# # Training loop
# for step in range(10000):
#   params, opt_state = train_step(params, opt_state, x0, timesteps, noise)
#   if step % 1000 == 0:
#     print(f'Step {step}: Loss: {loss_fn(params, x0, timesteps, noise)}')

# # Sampling from the model


# @jax.jit
# def ddim_sample(params, x_shape, timesteps):
#   x_t = jax.random.normal(rng, shape=x_shape)
#   for t in reversed(range(timesteps)):
#     noise = jax.random.normal(rng, shape=x_t.shape)
#     x_t = p_sample(lambda x, t: model.apply(params, x, t), x_t, t, noise)
#   return x_t


# # Generate new samples
# new_samples = ddim_sample(params, x0.shape, timesteps)
# print(f'Generated samples: {new_samples}')
