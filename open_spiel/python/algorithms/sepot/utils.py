import jax
import jax.numpy as jnp

import pyspiel

from open_spiel.python import policy
from open_spiel.python.algorithms import best_response
from open_spiel.python.algorithms import expected_game_score
from open_spiel.python.algorithms import get_all_states
from open_spiel.python.algorithms import sequence_form_lp

from open_spiel.python.algorithms.sepot import rnad_sepot as rnad
from open_spiel.python.algorithms.sepot.sepot import SePoT_RNaD


def take_policy_from_rnad(solver: rnad.RNaDSolver) -> policy.TabularPolicy:
  game = solver._game
  rnad_pols = policy.TabularPolicy(game)
  all_states = get_all_states.get_all_states(
    game,
    depth_limit=-1,
    include_terminals=False,
    include_chance_states=False,
    stop_if_encountered=False,
    to_string=lambda s: s.information_state_string())
  rollout = jax.vmap(solver.network.apply, (None, 0), 0)
  for iset in rnad_pols.state_lookup:
    state = all_states[iset]
    envs = [solver._state_as_env_step(state)]
    player = state.current_player()
    iset = state.information_state_string()
    tree_map = jax.tree_util.tree_map(lambda *e: jnp.stack(e, axis=0), *envs)
    pi, v, log_pi, logit = rollout(solver.params_target, jax.tree_util.tree_map(lambda *e: jnp.stack(e, axis=0), *envs))
    # pi = pi ...]
    state_policy = rnad_pols.policy_for_key(iset)
    # TODO: Change this to some form of broadcast
    for i in range(len(state_policy)):
        state_policy[i] = pi[0][i]
    pass
  return rnad_pols

def take_policy_from_rnad_at_once(solver: SePoT_RNaD) -> policy.TabularPolicy:
  game = solver._game
  rnad_pols = policy.TabularPolicy(game)
  all_states = get_all_states.get_all_states(
    game,
    depth_limit=-1,
    include_terminals=False,
    include_chance_states=False,
    stop_if_encountered=False,
    to_string=lambda s: s.information_state_string())

  rollout = jax.vmap(solver.network.apply, (None, 0), 0)
  env_steps = [solver._state_as_env_step(all_states[iset]) for iset in rnad_pols.state_lookup]
  pi, v, log_pi, logit = rollout(solver.params_target, jax.tree_util.tree_map(lambda *e: jnp.stack(e, axis=0), *env_steps))
  for iset_i, iset in enumerate(rnad_pols.state_lookup):
    state_policy = rnad_pols.policy_for_key(iset)
    for i in range(len(state_policy)):
      state_policy[i] = pi[iset_i, i]
  return rnad_pols
    

def take_policy_from_mvs(solver: SePoT_RNaD) -> policy.TabularPolicy:
  
  tab_policy = policy.TabularPolicy(solver.rnad._game)
  
  states_per_depth = []
  def traverse_tree(state, player, depth):
    if state.is_terminal():
      return
    if len(states_per_depth) <= depth:
      states_per_depth.append([])
    if state.current_player() == player:
      states_per_depth[depth].append(state)
    for action in state.legal_actions():
      new_s = state.clone()
      new_s.apply_action(action)
      traverse_tree(new_s, player, depth + 1)
  
  for pl in range(2):
    traverse_tree(solver.rnad._game.new_initial_state(), pl, 0)
    
    for depth, states_in_depth in enumerate(states_per_depth):
      for state in states_in_depth:
        if state.current_player() != pl or state.information_state_string() in solver.policy:
          continue
        avg_policy = solver.compute_policy(state, pl)
        # if depth == 0:
          # print(avg_policy)
        for iset, temp_policy in avg_policy.items():
          solver.policy[iset] = temp_policy
          tab_policy_pat = tab_policy.policy_for_key(iset)
          for action, prob in enumerate(temp_policy):
            tab_policy_pat[action] = prob
  return tab_policy

def resolve_first_subgame_then_rnad(solver: SePoT_RNaD) -> policy.TabularPolicy:
  rnad_policy = take_policy_from_rnad(solver.rnad)
  state = solver.rnad._game.new_initial_state()
  avg_policy = solver.compute_policy(state, 0)
  for iset, temp_policy in avg_policy.items():
    solver.policy[iset] = temp_policy
    print(temp_policy)
    tab_policy_pat = rnad_policy.policy_for_key(iset)
    for action, prob in enumerate(temp_policy):
      tab_policy_pat[action] = prob
  rnad_p1, rnad_p2 = evaluate_policy_both(solver.rnad._game, rnad_policy)
  print("RNAD, first resolve, exploitability P1: ", rnad_p1)
  print("RNAD, first resolve, exploitability P2: ", rnad_p2)
  

def evaluate_policy_both(game: pyspiel.Game, policy: policy.TabularPolicy) -> tuple[float, float]:
  return evaluate_policy_single(game, policy, 0), evaluate_policy_single(game, policy, 1)

def evaluate_policy_single(game: pyspiel.Game, policy: policy.TabularPolicy, player: int) -> float:
  assert player >= 0
  assert player < 2
  br = best_response.BestResponsePolicy(game, 1 - player, policy)
  return br.value(game.new_initial_state())
  

  
def compare_policies_mvs_rnad(solver: SePoT_RNaD):
  rnad_policy = take_policy_from_rnad(solver.rnad)
  mvs_policy = take_policy_from_mvs(solver)
  rnad_p1, rnad_p2 = evaluate_policy_both(solver.rnad._game, rnad_policy)
  mvs_p1, mvs_p2 = evaluate_policy_both(solver.rnad._game, mvs_policy)
  
  print("RNAD exploitability P1: ", rnad_p1)
  print("RNAD exploitability P2: ", rnad_p2)
  print("MVS exploitability P1: ", mvs_p1)
  print("MVS exploitability P2: ", mvs_p2)

def create_first_action_policy(game):
  policy = {}
  states = get_all_states(
        game,
        depth_limit=1000,
        include_terminals=False,
        include_chance_states=False,
        stop_if_encountered=False,
        to_string=lambda s: s.information_state_string())
  
  for state in states.values():
    legal_actions = state.legal_actions()
    retval = {action: 0 for action in legal_actions}
    retval[legal_actions[0]] = 1.
    policy[state.information_state_string()] = retval
  
  return
  