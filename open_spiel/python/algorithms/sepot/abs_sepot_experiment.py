import open_spiel.python.algorithms.sepot.rnad_sepot as rnad  
import open_spiel.python.algorithms.sepot.sepot as sepot
import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
parser.add_argument("--entropy_schedule", default=[1000, 10000], nargs="+", type=int, help="Entropy schedule")
parser.add_argument("--entropy_schedule_repeats", default=[50, 1], nargs="+", type=int, help="Entropy schedule repeats")
parser.add_argument("--rnad_network_layers", default=[256, 256], nargs="+", type=int, help="Network layers")
parser.add_argument("--mvs_network_layers", default=[256, 256], nargs="+", type=int, help="Network layers")
parser.add_argument("--transformation_network_layers", default=[256, 256], nargs="+", type=int, help="Network layers")
parser.add_argument("--learning_rate", default=3e-4, type=float, help="Learning Rate")
parser.add_argument("--c_vtrace", default=np.inf, type=float, help="Clipping of vtrace")
parser.add_argument("--rho_vtrace", default=np.inf, type=float, help="Clipping of vtrace")
parser.add_argument("--eta", default=0.2, type=float, help="Regularization term")
parser.add_argument("--num_transformations", default=10, type=int, help="Transformations of both players")

parser.add_argument("--iterations", default=100001, type=int, help="Amount of main iterations (each saves model)")
parser.add_argument("--save_each", default=5000, type=int, help="Length of each iteration in seconds")
parser.add_argument("--seed", default=42, type=int, help="Random seed")

def main():
  args = parser.parse_args([] if "__file__" not in globals() else None)

  game_name = "goofspiel"
  game_params = (
        ("num_cards", 3),
        ("imp_info", True),
        ("points_order", "descending")
  )

  max_trajectory = 4 
  rnad_config = rnad.RNaDConfig(
      game_name = game_name,
      game_params = game_params,
      trajectory_max =  max_trajectory,
      policy_network_layers = args.rnad_network_layers,
      mvs_network_layers = args.mvs_network_layers,
      transformation_network_layers = args.transformation_network_layers,
      
      batch_size = args.batch_size,
      learning_rate = args.learning_rate,
      entropy_schedule_repeats = args.entropy_schedule_repeats,
      entropy_schedule_size = args.entropy_schedule,
      c_vtrace = args.c_vtrace,
      rho_vtrace = args.rho_vtrace,
      eta_reward_transform = args.eta,

      num_transformations = args.num_transformations,
      matrix_valued_states = True,
      seed=  args.seed,
      state_representation = rnad.StateRepresentation.INFO_SET
  )
  
  sepot_config = sepot.SePoTConfig(
        rnad_config = rnad_config,
        resolve_iterations = 1000,
        subgame_size_limit = 10000000,
        subgame_depth_limit = 2,
        p = 0.5
    )
  sepot_solver = sepot.SePoT_RNaD(sepot_config)

  sepot_solver.compute_policy(sepot_solver.rnad._game.new_initial_state(), 0)

if __name__ == "__main__":
    main()