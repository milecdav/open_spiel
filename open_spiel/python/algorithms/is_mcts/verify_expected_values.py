import pyspiel
import pickle
import numpy as np

from open_spiel.python.algorithms.is_mcts.evaluator import ICMTSRNaDEvaluator
from open_spiel.python.algorithms.is_mcts.ismcts import ISMCTSBot, ChildSelectionPolicy, RootSelectionPolicy
from open_spiel.python.algorithms.sepot.rnad_sepot import RNaDSolver
from open_spiel.python.algorithms.sepot.utils import take_policy_from_rnad_at_once
from open_spiel.python.algorithms.best_response import BestResponsePolicy
from open_spiel.python.algorithms.expected_game_score import policy_value

import matplotlib.pyplot as plt
import os

# 
def verify_expected_values():
  file_name = "sepot_networks/goofspiel_5_descending/rnad_42_2000.pkl"
  player = 0
  
  with open(file=file_name, mode="rb") as f:
    solver = pickle.load(f)
  
  policy_from_rnad = take_policy_from_rnad_at_once(solver)
  
  evaluator = ICMTSRNaDEvaluator(solver, player, True)
  mcts_bot = ISMCTSBot(solver._game, evaluator, np.sqrt(2), 10000, random_state=np.random.RandomState(3332), child_selection_policy=ChildSelectionPolicy.UCT, root_selection_policy=RootSelectionPolicy.NETWORK)
  
  br_policy = BestResponsePolicy(solver._game, 1 - player, policy_from_rnad)
  
  folder = "sepot_plots/goofspiel_5_descending"
  if not os.path.exists(folder):
    os.makedirs(folder)
  # Goes through the game tree, when arriving to a player node, it first runs the MCTS to get the values of the children, then it computes the real expected values for each action (rnad vs br policy)
  def traverse_tree(state: pyspiel.State, player: int):
    if state.is_terminal():
      return
    if state.current_player() == player:
      mcts_bot.step(state)
      mcts_values = np.zeros(len(state.legal_actions_mask()))
      expected_values = np.zeros(len(state.legal_actions_mask()))
      for a, child in mcts_bot._root_node.child_info.items():
        mcts_values[a] = child.value()
      for root_probs, root_state in zip(mcts_bot._root_probs, mcts_bot._root_samples):
        for a in root_state.legal_actions():
          new_s = root_state.clone()
          new_s.apply_action(a)
          expected_values[a] += root_probs * policy_value(new_s, [policy_from_rnad, br_policy])[player]
          # expected_values[a] += root_probs * root_state.value()
      
      plt.plot([i for i in range(len(mcts_bot.action_vals))], mcts_bot.action_vals)
      plt.savefig(folder + "/" + state.information_state_string() + ".png")
      plt.cla()
      plt.close()
      # print(mcts_values)
      # print(expected_values)
      # print("----")
    for a in state.legal_actions():
      new_state = state.clone()
      new_state.apply_action(a)
      traverse_tree(new_state, player)
   
    
  traverse_tree(solver._game.new_initial_state(), player)


if __name__ == "__main__":
  verify_expected_values()