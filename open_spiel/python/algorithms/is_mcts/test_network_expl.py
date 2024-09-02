import torch
import pyspiel
import numpy as np
import pickle
import os

from open_spiel.python import policy
from open_spiel.python.algorithms.sepot.utils import evaluate_policy_both
import time
from open_spiel.python.algorithms import sequence_form_lp


def compute_nash(game):
  val1, val2, pi1, pi2 = sequence_form_lp.solve_zero_sum_game(game)
  goof5_policies = {"pi1": pi1, "pi2": pi2}
  folder = "nash_policies"
  if not os.path.exists(folder):
    os.makedirs(folder)
  with open("nash_policies/goof5_random.pkl", "wb") as f:
    pickle.dump(goof5_policies, f)
  return pi1, pi2

def get_rnad_policy(game, model):
  
  
  def extract_policy(pol, state):
    if state.is_terminal():
      return
    if not state.is_chance_node():
      tensor = torch.from_numpy(np.asarray(state.information_state_tensor(), dtype=np.float32))
      model_output = model(tensor).detach().numpy()
      pi, v = model_output[:-1], model_output[-1]
      pi = pi * np.asarray(state.legal_actions_mask(), dtype=np.float32)
      pi = pi / np.sum(pi)
      state_policy = pol.policy_for_key(state.information_state_string())
      
      for i in range(len(state_policy)):
        state_policy[i] = pi[i]
    for a in state.legal_actions():
      new_s = state.clone()
      new_s.apply_action(a)
      extract_policy(pol, new_s)
  
  rnad_pols = policy.TabularPolicy(game)
  state = game.new_initial_state()
  # state.apply_action(4)
  
  extract_policy(rnad_pols, state)
  return rnad_pols


def find_non_support_actions_in_rnad():
  game = pyspiel.load_game_as_turn_based("goofspiel", {"num_cards": 5, "points_order": "random", "imp_info": True})
  nash_pi1, nash_pi2 = compute_nash(game)
  
  model_base = "networks/goofspiel5_random/rnad_1947_0.pt"
  


def test_exploitability():
  model = torch.jit.load("networks/goofspiel5_random/rnad_1947_0.pt")
  game = pyspiel.load_game_as_turn_based("goofspiel", {"num_cards": 5, "points_order": "random", "imp_info": True})
  
  
  def extract_policy(pol, state):
    if state.is_terminal():
      return
    if not state.is_chance_node():
      tensor = torch.from_numpy(np.asarray(state.information_state_tensor(), dtype=np.float32))
      model_output = model(tensor).detach().numpy()
      pi, v = model_output[:-1], model_output[-1]
      pi = pi * np.asarray(state.legal_actions_mask(), dtype=np.float32)
      pi = pi / np.sum(pi)
      state_policy = pol.policy_for_key(state.information_state_string())
      
      for i in range(len(state_policy)):
        state_policy[i] = pi[i]
    for a in state.legal_actions():
      new_s = state.clone()
      new_s.apply_action(a)
      extract_policy(pol, new_s)
  
  rnad_pols = policy.TabularPolicy(game)
  state = game.new_initial_state()
  # state.apply_action(4)
  
  extract_policy(rnad_pols, state)
  # state = game.new_initial_state()
  # state.apply_action(2)
  # print(rnad_pols.policy_for_key(state.information_state_string()))
  exp1, exp2 = evaluate_policy_both(game, rnad_pols)
  print(exp1)
  print(exp2)
  


if __name__ == "__main__":
  start = time.time()
  game = pyspiel.load_game_as_turn_based("goofspiel", {"num_cards": 5, "points_order": "random", "imp_info": True})
  compute_nash(game)
  # with open("nash_policies/goof5_descending.pkl", "rb") as f:
  #   nash_policies = pickle.load(f)
  print(time.time() - start)