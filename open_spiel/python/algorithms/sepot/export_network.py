import pickle

import jax.numpy as jnp
from open_spiel.python.algorithms.sepot.export_network_class import RNaDSmall

def export_network_to_smaller(file_name):
  with open(file_name, "rb") as f:
    network = pickle.load(f)
  
  state = network._game.new_initial_state()
  
  
  rnad_small = RNaDSmall(file_name, network.params_target, network.network.out_dims, network.network.hidden_dims, network.network.residual_blocks, network._np_rng.get_state(), network.config.game_name, network.config.game_params)
  
  new_name = file_name.replace(".pkl", "_small.pkl")
  with open(new_name, "wb") as f:
    pickle.dump(rnad_small, f)
    
  with open(new_name, "rb") as f:
    resaved_network = pickle.load(f)
  
  env_step = network._state_as_env_step(state)
  pi1 = network._network_jit_apply(network.params_target, env_step)
  pi2 = rnad_small.get_policy(state)
  pi3 = resaved_network.get_policy(state)
  
  # print(pi1[pi1 > 0.01])
  # print(pi2[pi2 > 0.01])
  # print(pi3[pi3 > 0.01])
  print(jnp.allclose(pi1, pi2))
  print(jnp.allclose(pi1, pi3))
  print(jnp.allclose(pi2, pi3))
  
   


if __name__ == "__main__":
  export_network_to_smaller("sepot_networks/dark_chess/rnad_666321_400000.pkl")
  export_network_to_smaller("sepot_networks/battleship_7x7_4s3s3s2/rnad_987456_285000.pkl")
  pass