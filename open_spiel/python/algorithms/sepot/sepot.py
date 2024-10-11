import pyspiel
import chex
import jax
import jax.numpy as jnp
import numpy as np


from open_spiel.python.algorithms.sepot.rnad_sepot import RNaDConfig, RNaDSolver
from open_spiel.python.algorithms.sepot.cfr_sepot import SePoTCFR
from open_spiel.python.algorithms.reconstruct_public_state import reconstruct_states


def actions_per_player(state: pyspiel.State, player: int):
  a = 0
  for h in state.full_history():
    if h.player == player:
      a += 1
  return a


@chex.dataclass(frozen=True)
class SePoTConfig:
  rnad_config: RNaDConfig
  resolve_iterations: int
  subgame_size_limit: int
  subgame_depth_limit: int
  p: float

class SePoT_RNaD:
  def __init__(self, sepot_config: SePoTConfig) -> None:
    self.rnad = RNaDSolver(sepot_config.rnad_config)
    self.config = sepot_config


    self.policy = {}

  def fixed_policy(self, state):
    policy = state.legal_action_mask()
    return policy / np.sum(policy)

  def reset_policy(self):
    self.policy = {}
    
  def train(self, num_iterations: int):
    for i in range(num_iterations):
      self.rnad.step()

  def compute_policy(self, state: pyspiel.State, player: int):
    histories, use_search = reconstruct_states(state)
    if not use_search:
      return self.rnad.action_probabilities(state)
    states, counterfactual_values, reaches, fixed_reaches = self.reconstruct_public_belief_state(histories, player)
    
    create_gadget = actions_per_player(state, 1 - player) > 0
    
    subgame_solver = SePoTCFR(self, states, counterfactual_values, reaches[player], reaches[-1], fixed_reaches, player, self.config.subgame_depth_limit, create_gadget, self.config.p)
    
    subgame_solver._alternating_updates = False
    # subgame_solver.multiple_steps(self.config.resolve_iterations)
    subgame_solver.multiple_steps(self.config.resolve_iterations)
    
    policy = subgame_solver.average_policy_dict(player)
    
    return policy

  def reconstruct_public_belief_state(self, histories: list[list[int]], player: int):
    states = []
    reaches_per_state = [[], [], []]
    counterfactual_values = [{}, {}]
    counterfactual_reaches = [{}, {}]
    fixed_reaches_per_state = []
    # Go through each history and get policy from network, then follow it u
    for history in histories:
      state = self.rnad._game.new_initial_state()
      reaches = [1.0, 1.0, 1.0]
      fixed_reach = 1.0
      for a in history:
        if state.is_chance_node():
          reaches[-1] *= state.chance_outcomes()[a]
        else:
          assert not state.is_terminal()

          # Checks if policy was computed, otherwise it uses the policy from rnad
          if state.information_state_string() in self.policy:
            policy = self.policy[state.information_state_string()]
          else:
            if state.current_player() == 1 - player:
              policy = self.rnad.action_probabilities(state)
              # TODO: Ask Ondra if the actions size is all actions or only the legal ones
              fixed_policy = self.fixed_policy(state)
              fixed_policy = fixed_policy / jnp.sum(fixed_policy)
              fixed_reach *= fixed_policy[a]
            else:
              assert False
            
          reaches[state.current_player()] *= policy[a]
        state.apply_action(a)
      # First value is identity transformation
      expected_value = self.rnad.get_multi_valued_states(state)[0]
      for pl in range(2):
        iset = state.information_state_string(pl)
        if iset not in counterfactual_values[pl]:
          counterfactual_values[pl][iset] = 0.0
          counterfactual_reaches[pl][iset] = 0.0
        counterfactual_values[pl][iset] += expected_value * reaches[1 - pl] * reaches[-1]
        counterfactual_reaches[pl][iset] += reaches[1 - pl] * reaches[-1]
      states.append(state)
      for i in range(len(reaches)):
        reaches_per_state[i].append(reaches[i])
      fixed_reaches_per_state.append(fixed_reach)
      # reaches_per_state.append(reaches)
    
    # Normalizing in root
    for pl in range(2):
      for iset in counterfactual_values[pl]:
        if counterfactual_reaches[pl][iset] >= 1e-15:
          counterfactual_values[pl][iset] /= counterfactual_reaches[pl][iset]

    counterfactual_values_per_state = []
    for state in states:
      iset = state.information_state_string(1 - player)
      counterfactual_values_per_state.append(counterfactual_values[1 - player][iset])

    assert len(states) == len(counterfactual_values_per_state)
    for rs in reaches_per_state:
      assert len(states) == len(rs)
    assert len(fixed_reaches_per_state) == len(states)
    return states, counterfactual_values_per_state, reaches_per_state, fixed_reaches_per_state
    