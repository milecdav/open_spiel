
import os
import argparse
import numpy as np
import pickle

import time

import open_spiel.python.algorithms.sepot.rnad_sepot as rnad

from open_spiel.python.algorithms.get_all_states import get_all_states
from pyinstrument import Profiler
import pyspiel
# from open_spiel.python.algorithms.

parser = argparse.ArgumentParser()
# Experiments specific arguments

parser.add_argument("--game_simulations", default=300, type=int, help="Amount of main iterations (each saves model)")
parser.add_argument("--model_path", default="sepot_networks/dark_chess/rnad_666321", type=str, help="Length of each iteration in seconds")
parser.add_argument("--iterations_range", default=[200000, 210000, 5000], nargs="+", type=int, help="Ship sizes")
 
parser.add_argument("--seed", default=42, type=int, help="Random seed")


def evaluate_network():
  """Evaluates a network by playing it against random player."""
  args = parser.parse_args([] if "__file__" not in globals() else None)
  assert len(args.iterations_range) == 3
  np_rng = np.random.RandomState(args.seed)
  # results = [[], []]
  for i in range(*args.iterations_range):
    model_path = args.model_path + "_" + str(i) + ".pkl"
    with open(model_path, "rb") as f:
      network = pickle.load(f)
    for player in range(2): # For each player
      result = []
      for _ in range(args.game_simulations):
        state = network._game.new_initial_state()
        while not state.is_terminal(): 
          if state.current_player() == player:
            aps = network.action_probabilities(state)
            actions, probabilities = [], []
            for action, probs in aps.items():
              actions.append(action)
              probabilities.append(probs)
            action = np_rng.choice(actions, p=probabilities)
          else:
            action = np_rng.choice(state.legal_actions())
          state.apply_action(action)
        returns = state.returns()
        result.append(returns[player])
      print("Iteration", i, ";Player", player, ";mean:", np.mean(result), ";std:", np.std(result), flush=True)
  
  # for _ in range(num_games):
  #   state = game.new_initial_state()
  #   while not state.is_terminal():
  #     current_player = state.current_player()
  #     legal_actions = state.legal_actions()
  #     action = network.sample_action(state)
  #     state.apply_action(action)
  #   returns = state.returns()
  #   for player in range(num_players):
  #       outcomes[returns[player], player] += 1
  # return outcomes / num_games



if __name__ == "__main__":
  
  evaluate_network()