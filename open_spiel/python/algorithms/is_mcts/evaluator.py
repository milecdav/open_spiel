import torch
import numpy as np
import time
import jax
import jax.numpy as jnp
import functools


from open_spiel.python.policy import TabularPolicy

from open_spiel.python.algorithms.mcts import Evaluator
from open_spiel.python.algorithms.mcts_agent import MCTSAgent
from open_spiel.python.algorithms.is_mcts.ismcts import ISMCTSBot, ChildSelectionPolicy
from open_spiel.python.jax.cfr.jax_cfr import JaxCFR
from open_spiel.python.algorithms.sepot.utils import evaluate_policy_both
from open_spiel.python.algorithms.is_mcts.test_network_expl import get_rnad_policy
from open_spiel.python.algorithms.sepot.rnad_sepot import RNaDSolver


import pyspiel
import pickle
import enum
from open_spiel.python.algorithms.sequence_form_lp import solve_zero_sum_game

# class RnadType(enum.Enum):
  

class ICMTSRNaDEvaluator(Evaluator):
  def __init__(self,
               rnad: RNaDSolver,
               player: int,
               keep_cache: bool = False) -> None:
    self._rnad = rnad
    self._player = player
    self._keep_cache = keep_cache
    self._cache = {}
  
  def evaluate(self, state):
    """Returns evaluation on given from the infoset network."""
    assert self._player >= 0
    envs = [self._rnad._state_as_env_step(state)]  
    _, v, _, _ = self.call_network(envs)
    v = np.asarray(v[0])
    if state.current_player() == 0:
      return v, -v
    else:
      return -v, v
    
  def prior(self, state):
    """For player that is acting it just outputs the policy of the network, for the other player it outputs uniform policy."""
    assert self._player >= 0
    policy = self.get_policy(state)
    policy = [(a, policy[a]) for a in state.legal_actions()]
    return policy

  def get_policy(self, state):
    """For player that is acting it just outputs the policy of the network, for the other player it outputs uniform policy."""
    assert self._player >= 0
    
    if self._keep_cache and state.information_state_string() in self._cache:
      return self._cache[state.information_state_string()]
    
    envs = [self._rnad._state_as_env_step(state)]  
    # pi = pi[0] / jnp.sum(pi[0])
    pi, _, _, _ = self.call_network(envs)
    pi = np.asarray(pi[0])
    pi = pi / np.sum(pi)
    
    if self._keep_cache:
      self._cache[state.information_state_string()] = pi
      
    return pi
  
  @functools.partial(jax.jit, static_argnums=(0,))
  def call_network(self, env_state):
    rollout = jax.vmap(self._rnad.network.apply, (None, 0), 0)
    return rollout(self._rnad.params_target, jax.tree_util.tree_map(lambda *e: jnp.stack(e, axis=0), *env_state))
  
  
class Resampler():
  def __init__(self) -> None:
    iset_map = {}
  
  def __call__(self, state: pyspiel.State, player: int):
    assert state.current_player() == player
    iset = state.information_state_string()
    
  

def test():
  random_state = np.random.seed(997)
  game = pyspiel.load_game_as_turn_based("goofspiel", {"num_cards": 5, "points_order": "descending", "imp_info": True})
  # print(game.get_type().long_name)
  solver=  JaxCFR(game)
  solver.multiple_steps(2000)
  eq_policy = solver.average_policy()
  # v1,v2,pi1,pi2 = solve_zero_sum_game(game)
  evaluator = ISMCTSPruneEvaluator("networks/goofspiel5_descending/rnad_12964_142000.pt", 0)
  model = torch.jit.load("networks/goofspiel5_descending/rnad_12964_142000.pt")
  # eq_policy = get_rnad_policy(game, model)
  
  
  bot = ISMCTSBot(game, evaluator, np.sqrt(2), 50000, random_state=random_state, child_selection_policy=ChildSelectionPolicy.UCT)
  bot._policy = eq_policy
  
  visited_iset = set()
  policy = TabularPolicy(game)
  
  def traverse_tree(state, p1_reach, p2_reach):
    if state.is_terminal():
      return
    if not state.is_chance_node() and ((state.current_player() == 0 and p1_reach >= 1e-5) or (state.current_player() == 1 and p2_reach >= 1e-5) ) and state.current_player() == 0:
      iset = state.information_state_string()
      state_eq_policy = eq_policy.action_probabilities(state)
      if iset not in visited_iset:
        visited_iset.add(iset)
        bot.step(state)
        node = bot.lookup_or_create_node(state)
        values = (np.asarray(state.legal_actions_mask(), dtype=np.float32) - 1) * 1000 # Invalid actions become -1000
        for a, child in node.child_info.items():
          values[a] += child.value()
        # print(values)
        mask = values >= (np.max(values) - 0.06)
        state_policy = np.zeros_like(mask, dtype=np.float32)
        state_eq_policy = eq_policy.action_probabilities(state)
        for a, prob in state_eq_policy.items():
          state_policy[a] = prob * mask[a]
        # print(state_policy)
        # print(state_eq_policy)
        state_policy = state_policy / np.sum(state_policy)
        my_policy = policy.policy_for_key(iset)
        # TODO: Change this to some form of broadcast
        for i in range(len(state_policy)):
          my_policy[i] = state_policy[i]
        # print(mask)
        # print(state_eq_policy)
        # print(my_policy)
    # return
    for a in state.legal_actions():
      new_p1_reach = p1_reach
      new_p2_reach = p2_reach
      if state.current_player() >= 0:
        new_p1_reach = p1_reach if state.current_player() == 1 else p1_reach * eq_policy.action_probabilities(state)[a]
        new_p2_reach = p2_reach if state.current_player() == 0 else p2_reach * eq_policy.action_probabilities(state)[a]
      new_s = state.clone()
      new_s.apply_action(a)
      traverse_tree(new_s, new_p1_reach, new_p2_reach)
        
        
  start = time.time()
  state = game.new_initial_state()
  traverse_tree(state, 1.0, 1.0)
  print(time.time() - start)
  print("Equilibrium", evaluate_policy_both(game, eq_policy))
  print("MCTS", evaluate_policy_both(game, policy))
  
  # state.apply_action(3)
  # for i in range(1):
  #   state = game.new_initial_state()
  #   while not state.is_terminal():
  #     action = bot.step(state)
  #     state.apply_action(action)
  
  
def game_play_test():
  
  file_name = "sepot_networks/battleship_5x5_3s2s2/rnad_7357_0.pkl"
  evaluator1 = ICMTSRNaDEvaluator(file_name, 0)
  evaluator2 = ICMTSRNaDEvaluator(file_name, 1)
  np_rng1 = np.random.RandomState(1234)
  np_rng2 = np.random.RandomState(1542)
  jnp_rng = jax.random.PRNGKey(1234)
  np_sample_action = np.random.RandomState(3472)
  p1_bot = ISMCTSBot(evaluator1.rnad._game, evaluator1, np.sqrt(2), 5000, random_state=np_rng1, child_selection_policy=ChildSelectionPolicy.UCT)
  p2_bot = ISMCTSBot(evaluator2.rnad._game, evaluator2, np.sqrt(2), 5000, random_state=np_rng2, child_selection_policy=ChildSelectionPolicy.UCT)

  use_p1_bot = True
  use_p2_bot = False

  overall_reward = 0.0
  for i in range(1000):
    if i % 50 == 0:
      print(i)
    state = evaluator1.rnad._game.new_initial_state()
    while not state.is_terminal():
      if not state.is_chance_node():
        policy = evaluator1.get_policy(state)
        jnp_rng, now_rng = jax.random.split(jnp_rng)
        action = jax.random.choice(now_rng, len(policy), p=policy)
        
        if state.current_player() == 0:
          policy = evaluator1.get_policy(state)
          
          action = jax.random.choice(jnp_rng, len(policy), p=policy)
          if use_p1_bot:
            action = p1_bot.step(state)
            
        elif state.current_player() == 1:
          policy = evaluator2.get_policy(state)
          
          action = jax.random.choice(jnp_rng, len(policy), p=policy)
          if use_p2_bot:
            action = p2_bot.step(state)
        # print(policy)
        state.apply_action(action)
    # overall_reward += state.returns()[0]
    print(state.returns()[0])
    # v = 
    overall_reward += state.returns()[0]
  print(overall_reward)
    
    


if __name__ == '__main__':
  game_play_test()