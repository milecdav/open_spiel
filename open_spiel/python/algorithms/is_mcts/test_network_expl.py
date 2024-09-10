import torch
import pyspiel
import numpy as np
import pickle
import os

from open_spiel.python import policy
from open_spiel.python.jax.cfr.jax_cfr import JaxCFR
from open_spiel.python.algorithms.sepot.utils import evaluate_policy_both, take_policy_from_rnad, take_policy_from_rnad_at_once
import time
from open_spiel.python.algorithms import sequence_form_lp
import matplotlib.pyplot as plt


def compute_nash(game):
  jax_cfr = JaxCFR(game)
  jax_cfr.multiple_steps(5000)
  avg_policy = jax_cfr.average_policy()
  folder = "nash_policies"
  if not os.path.exists(folder):
    os.makedirs(folder)
  with open("nash_policies/goof5_random.pkl", "wb") as f:
    pickle.dump(avg_policy, f)
  return avg_policy


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
  # nash_pi1, nash_pi2 = compute_nash(game)
  with open("nash_policies/goof5_random.pkl", "rb") as f:
    nash_policies = pickle.load(f)
  model_base = "networks/goofspiel5_random/rnad_1947_"
  models = [torch.jit.load(model_base + str(i) + ".pt") for i in range(0, 100001, 3000)]
  in_support = [0 for i in models]
  smallest_possible = 5e-4
  
  all_used_infosets = set()
  
  def extract_policy(state, p1_reach, p2_reach):
    if state.is_terminal():
      return
    if not state.is_chance_node():
      reach = p1_reach if state.current_player() == 0 else p2_reach
      if state.information_state_string() not in all_used_infosets and reach > 1e-6:
        all_used_infosets.add(state.information_state_string())
        tensor = torch.from_numpy(np.asarray(state.information_state_tensor(), dtype=np.float32))
        for model_id, model in enumerate(models):
          model_output = model(tensor).detach().numpy()
          pi, v = model_output[:-1], model_output[-1]
          pi = pi * np.asarray(state.legal_actions_mask(), dtype=np.float32)
          pi = pi / np.sum(pi)
          state_policy = nash_policies.policy_for_key(state.information_state_string())
          for i in range(len(state_policy)):
            if state_policy[i] < smallest_possible and pi[i] > smallest_possible:
              in_support[model_id] += 1
    for a in state.legal_actions():
      # p1_reach = p1_reach if state.current_player() != 0 else p1_reach * state_policy[a]
      # p2_reach = p2_reach if state.current_player() != 1 else p2_reach * state_policy[a]
      new_s = state.clone()
      new_s.apply_action(a)
      extract_policy(new_s, p1_reach, p2_reach)
      
  state = game.new_initial_state()
  extract_policy(state, 1.0, 1.0)
  print(in_support)


def test_exploitability():
  model = torch.jit.load("networks/goofspiel5_random/rnad_1947_0.pt")
  game = pyspiel.load_game_as_turn_based("goofspiel", {"num_cards": 5, "points_order": "random", "imp_info": True})
  
  
  def extract_policy(pol, state, p1_reach, p2_reach):
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
      p1_reach = p1_reach if state.current_player() != 0 else p1_reach * state_policy[a]
      p2_reach = p2_reach if state.current_player() != 1 else p2_reach * state_policy[a]
      new_s = state.clone()
      new_s.apply_action(a)
      extract_policy(pol, new_s, p1_reach, p2_reach)
  
  
  rnad_pols = policy.TabularPolicy(game)
  state = game.new_initial_state()
  # state.apply_action(4)
  
  extract_policy(rnad_pols, state, 1.0, 1.0)
  # state = game.new_initial_state()
  # state.apply_action(2)
  # print(rnad_pols.policy_for_key(state.information_state_string()))
  exp1, exp2 = evaluate_policy_both(game, rnad_pols)
  print(exp1)
  print(exp2)
  
#  [38405, 38405, 38404, 38405, 38403, 38401, 38403, 38402, 38396, 38394, 38398, 38378, 38397, 38404, 38399, 38393, 38390, 38397, 38387, 38392, 38388, 38351, 38346, 38372, 38354, 38383, 38307, 38382, 38306, 38312, 38303, 38338, 38361, 38322, 38322, 38306, 38161, 38232, 38296, 38227, 38317, 38287, 38116, 38223, 38190, 38123, 38107, 38236, 38243, 38179, 38315, 38323, 38169, 38299, 38310, 38292, 38316, 38310, 38347, 38310, 38335, 38304, 38308, 38325, 38356, 38358, 38359, 38318, 38351, 38267, 38330, 38348, 38343, 38305, 38323, 38334, 38332, 38330, 38306, 38204, 38302, 38297, 38255, 38304, 38004, 38173, 38233, 38309, 38217, 38306, 38305, 38326, 38284, 38279, 38289, 38314, 38295, 38317, 38196, 38245, 38244]


def compute_exploitability_multiple_nets():
  networks_folder = "sepot_networks/"
  experiment_type = [
    # Name, folder, seed, iters
    ("leduc_small_network", "leduc/", 1234987, 100001),
    ("leduc_large_network", "leduc/", 88442211, 100001),
    ("battleship 2x2 small network", "battleship_2x2_2/", 1234987, 100001),
    ("battleship 2x2 large network", "battleship_2x2_2/", 666789, 100001),
    ("battleship 3x2 small network", "battleship_3x2_2s2/", 1234987, 100001),
    ("battleship 3x2 large network", "battleship_3x2_2s2/", 666789, 100001),
  ]
  
  
  
  for experiment_name, folder, seed, max_iter in experiment_type:
    
    folder = networks_folder + folder
    iters = [i for i in range(0, max_iter, 200)]
    policies = []
    for i in iters:
      file_name = folder + "rnad_" + str(seed) + "_" + str(i) + ".pkl"
      with open(file_name, "rb") as f:
        solver = pickle.load(f)
    
      policies.append(take_policy_from_rnad_at_once(solver))
    print(experiment_name)
    exploits = [evaluate_policy_both(solver._game, p) for p in policies]
    print(exploits)
    sepot_plots = "sepot_plots/" + experiment_name + "/" 
    if not os.path.exists(sepot_plots):
      os.makedirs(sepot_plots)
    for iset_i, iset in enumerate(policies[0].state_lookup):
      plot_name = sepot_plots + iset + "_" + str(seed) + ".png"
      # policy = 
      plt.plot(iters, [p.policy_for_key(iset) for p in policies])
      plt.savefig(plot_name)
      plt.cla()
      plt.close()
      
    cfr = JaxCFR(solver._game)
    cfr.multiple_steps(2000)
    cfr_policy = cfr.average_policy()
    for iset_i, iset in enumerate(cfr_policy.state_lookup):
      plot_name = sepot_plots + iset + ".png"
      # policy = 
      plt.plot([0, max_iter], [cfr_policy.policy_for_key(iset), cfr_policy.policy_for_key(iset)])
      plt.savefig(plot_name)
      plt.cla()
      plt.close()
    
  


if __name__ == "__main__":
  # game = pyspiel.load_game_as_turn_based("goofspiel", {"num_cards": 5, "points_order": "random", "imp_info": True})
  # compute_nash(game)
  # with open("nash_policies/goof5_random.pkl", "rb") as f:
  #   nash_policies = pickle.load(f)
    
  start = time.time()
  
  compute_exploitability_multiple_nets()
  print(time.time() - start)
