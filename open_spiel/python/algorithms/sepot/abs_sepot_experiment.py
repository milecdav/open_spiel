import open_spiel.python.algorithms.sepot.rnad_sepot as rnad  
import open_spiel.python.algorithms.sepot.sepot as sepot

def main():
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
      policy_network_layers = 0,
      mvs_network_layers = 0,
      transformation_network_layers = 0,
      
      batch_size = 0,
      learning_rate = 0,
      entropy_schedule_repeats = 0,
      entropy_schedule_size = 0,
      c_vtrace = 0,
      rho_vtrace = 0,
      eta_reward_transform = 0,

      num_transformations = 0,
      matrix_valued_states = True,
      seed = 0
  )
  
  sepot_config = sepot.SePoTConfig(
        rnad_config = rnad_config,
        resolve_iterations = 1000,
        subgame_size_limit = 10000000,
        subgame_depth_limit = 2)
  sepot_solver = sepot.SePoT_RNaD(sepot_config)

  sepot_solver.compute_policy(sepot_solver.rnad._game)

if __name__ == "__main__":
    main()