import argparse
import os
import pyspiel
import time
import pickle
import numpy as np
import open_spiel.python.algorithms.rnad.rnad as rnad

parser = argparse.ArgumentParser()
# Experiments specific arguments
parser.add_argument("--evaluate", default=False, type=bool, help="Evaluate the model")
parser.add_argument("--evaluate_each", default=1000, type=int, help="Evaluate the model after that many iterations")
parser.add_argument("--evaluate_last", default=False, type=bool, help="Evaluate the model after last iteration")
parser.add_argument("--iterations", default=20, type=int, help="Amount of main iterations (each saves model)")
parser.add_argument("--save_each", default=1000, type=int, help="Length of each iteration in seconds")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--model_path", default="", type=str, help="Path to the policy model, if should be loaded")

# (Se)RNaD experiment specific arguments
parser.add_argument("--batch_size", default=16, type=int, help="Batch size")
parser.add_argument("--entropy_schedule", default=(500, 10000), type=tuple, help="Entropy schedule")
parser.add_argument("--entropy_schedule_repeats", default=(200, 1), type=tuple, help="Entropy schedule repeats")
parser.add_argument("--network_layers", default=(1024, 1024), type=tuple, help="Network layers")
parser.add_argument("--learning_rate", default=3e-4, type=float, help="Learning Rate")
parser.add_argument("--c_vtrace", default=1.0, type=float, help="Clipping of vtrace")
parser.add_argument("--rho_vtrace", default=2.0, type=float, help="Clipping of vtrace")
parser.add_argument("--eta", default=0.2, type=float, help="Regularization term")
parser.add_argument("--epsilon", default=0.0, type=float, help="Part of uniform policy in sampling policy")
parser.add_argument("--p1_transformations", default=10, type=int, help="Transformations of P1")
parser.add_argument("--p2_transformations", default=10, type=int, help="Transformations of P2")
 
from pyinstrument import Profiler

def train():
  args = parser.parse_args([] if "__file__" not in globals() else None)
  game_params = tuple()
  game_name ="dark_chess"
  path = "void_networks/dark_chess/"
  if not os.path.exists(path):
    os.makedirs(path)
  
  # print(game.max_game_length())
  # return

  max_trajectory = 200
  if args.model_path:
    with open(args.model_path, "rb") as f:
      solver = pickle.load(f)
      
    i = int(args.model_path.split(".")[-2].split("_")[-1]) + 1
  else:
    config = rnad.RNaDConfig(
        game_name = game_name, 
        game_params = game_params,
        trajectory_max = max_trajectory,
        policy_network_layers = args.network_layers,
        batch_size = args.batch_size,
        learning_rate = args.learning_rate,
        entropy_schedule_repeats = args.entropy_schedule_repeats,
        entropy_schedule_size = args.entropy_schedule,
        c_vtrace = args.c_vtrace if args.c_vtrace >= 0 else np.inf,
        rho_vtrace = args.rho_vtrace if args.rho_vtrace >= 0 else np.inf,
        eta_reward_transform = args.eta,
        seed=  args.seed,
        state_representation = rnad.StateRepresentation.OBSERVATION
        
    )
    solver = rnad.RNaDSolver(config)
    i = 0
   
  start = time.time()
  print_iter_time = time.time() # We will save the model in first step
  # profiler = Profiler()
  # profiler.start()
  for iteration in range(i, args.iterations + i):
    solver.step()
    # print(iteration, flush=True)
    if iteration % args.save_each == 0:
        
      file = "rnad_" + str(args.seed) + "_" + str(iteration) + ".pkl"
      file_path = path + file
      with open(file_path, "wb") as f:
        pickle.dump(solver, f)
      print("Saved at iteration", iteration, "after", int(time.time() - start), flush=True)

    # Prints time each hour
    if time.time() > print_iter_time:
      print("Iteration ", iteration, flush=True)

      print_iter_time = time.time() + 60 * 60
  # profiler.stop()
  # print(profiler.output_text(color=True, unicode=True))
  i+= 1 
    
def play(saved_model):
  args = parser.parse_args([] if "__file__" not in globals() else None)
  np.random.seed(args.seed)
  game_params = tuple()
  game_name ="dark_chess"
  path = "void_networks/dark_chess/"
  if not os.path.exists(path):
    os.makedirs(path)
  
  with open(saved_model, "rb") as f:
    solver = pickle.load(f)
  
  game = pyspiel.load_game(game_name)

  max_length = 0
  lengths = []
  for i in range(1000):
    length = 0
    state = game.new_initial_state()
    while not state.is_terminal():
      policy = solver.action_probabilities(state)
      policy_vector = np.zeros(game.num_distinct_actions())
      for i, p in policy.items():
        policy_vector[i] = p
      action = np.random.choice(game.num_distinct_actions(), p=policy_vector)
      state.apply_action(action)
      length += 1
    lengths.append(length)
    max_length = max(max_length, length)
  lengths.sort()
  print(max_length)
  for i in range(0, 1000, 99):
    print(i, lengths[i])
    # print(action)



    

    
if __name__ == "__main__":
  train()