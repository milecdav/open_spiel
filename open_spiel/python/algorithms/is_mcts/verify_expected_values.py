import pyspiel
import pickle
import numpy as np

from open_spiel.python.algorithms.is_mcts.evaluator import ICMTSRNaDEvaluator
from open_spiel.python.algorithms.is_mcts.ismcts import ISMCTSBot, ChildSelectionPolicy, RootSelectionPolicy
from open_spiel.python.algorithms.sepot.rnad_sepot import RNaDSolver
from open_spiel.python.algorithms.sepot.utils import take_policy_from_rnad_at_once
from open_spiel.python.algorithms.best_response import BestResponsePolicy
from open_spiel.python.algorithms.expected_game_score import policy_value


def verify_expected_values():
  file_name = "sepot_networks/goofspiel_5_descending/rnad_42_2000.pkl"
  player = 0
  
  with open(file=file_name, mode="rb") as f:
    solver = pickle.load(f)
  
  policy_from_rnad = take_policy_from_rnad_at_once(solver)
  
  evaluator = ICMTSRNaDEvaluator(solver, player, True)
  mcts_bot = ISMCTSBot(solver._game, evaluator, np.sqrt(4), 50000, random_state=np.random.RandomState(3332), child_selection_policy=ChildSelectionPolicy.UCT, root_selection_policy=RootSelectionPolicy.PUCT)
  
  br_policy = BestResponsePolicy(solver._game, 1, policy_from_rnad)
  
  
  
  def traverse_tree(state: pyspiel.State):
    if state.is_terminal():
      return
    if not state.is_chance_node():
      pass
   
    
    


if __name__ == "__main__":
  verify_expected_values()