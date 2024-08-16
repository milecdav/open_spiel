from open_spiel.python.algorithms.sepot import rnad_sepot_parallel as rnad

import numpy as np
import jax
import os

from multiprocessing.managers import BaseManager
import multiprocessing as mp
import time


class RandomClass():
  def __init__(self):
    self.value = {}
  
  def set(self, value):
    self.value = value
    
    
  def get(self):
    return self.value


def just_sleep(params, queuem):
  print(params.get())
  time.sleep(20)
  print(params.get())

def test_parallel(): 
  game_name = "dark_chess"  
  
  max_trajectory = 80
  rnad_config = rnad.RNaDConfig(
      game_name = game_name, 
      game_params = tuple(),
      trajectory_max =  max_trajectory,
      policy_network_layers = [32, 32],
      mvs_network_layers = [32, 32],
      transformation_network_layers = [32, 32],
      
      batch_size = 8,
      learning_rate = 3e-4,
      entropy_schedule_repeats = [100, 1],
      entropy_schedule_size = [2000, 10000],
      c_vtrace = 1.5,
      rho_vtrace = np.inf,
      eta_reward_transform = 0.2,

      num_transformations = 15,
      matrix_valued_states = True,
      seed = 333,
      state_representation = rnad.StateRepresentation.OBSERVATION
  )
  i = 0

  solver =  rnad.RNaDSolver(rnad_config)
  
  BaseManager.register('RandomClass', RandomClass)
  manager = BaseManager()
  
  manager.start()
  r_class = manager.RandomClass()
  r_class.set(solver.params)
  
  
  processes = [mp.Process(target=just_sleep, args=(r_class,)) for _ in range(4)]
  for p in processes:
    p.daemon = True
    p.start()
  print("before step")
  solver.step()
  r_class.set(solver.params)
  print("after step")
  time.sleep(20)
  
  
  pass

def sum_worker(shared_num1, shared_num2, shared_list):
  while True:
    # Access the shared values and calculate the sum
    num1 = shared_num1.value
    num2 = shared_num2.value
    result = num1 + num2
    
    # Append the result to the shared list
    shared_list.append(result)
    
    # Sleep for a short period to simulate work being done
    time.sleep(1)

def update_numbers(shared_num1, shared_num2, new_num1, new_num2):
  # Update the shared values
  shared_num1.value = new_num1
  shared_num2.value = new_num2
    
def basic_parallel():
  manager = mp.Manager()
    
  # Shared values using multiprocessing.Value
  shared_num1 = mp.Value('i', 1)
  shared_num2 = mp.Value('i', 2)
  
  # Shared list using Manager
  shared_list = manager.list()

  processes = [mp.Process(target=sum_worker, args=(shared_num1, shared_num2, shared_list)) for _ in range(4)]
  
  for p in processes:
    p.daemon = True
    p.start()
    
  
  time.sleep(2.2)
  shared_num1.value = 3
  time.sleep(2.2)
  shared_num2.value = 4
  time.sleep(2.2)
  for i in shared_list:
    print(i)
    
import time    
    
def goofspiel_parallel():
  os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
  game_name = "goofspiel"
  game_params = (
        ("num_cards", 5),
        ("imp_info", True),
        ("points_order", "descending")
  )
  test_iters = 2000
  max_trajectory = (5 - 1) * 2 
  rnad_config = rnad.RNaDConfig(
      game_name = game_name, 
      game_params = game_params,
      trajectory_max =  max_trajectory,
      policy_network_layers = [32, 32],
      mvs_network_layers = [32, 32],
      transformation_network_layers = [32, 32],
      
      batch_size = 32,
      learning_rate = 3e-4,
      entropy_schedule_repeats = [100, 1],
      entropy_schedule_size = [500, 2000],
      c_vtrace = 1.0,
      rho_vtrace = 1.0,
      eta_reward_transform = 0.2,

      num_transformations = 10,
      matrix_valued_states = True,
      seed= 4444
  )
  solver = rnad.RNaDSolver(rnad_config)
  print(jax.devices('cpu'))
  # print(jax.devices('gpu'))
  
  
  start = time.time()
  solver.parallel_steps(test_iters)
  print(time.time() - start)
  
  
  start = time.time()
  for i in range(test_iters):
    solver.step()
  print(time.time() - start)
    
    
def dark_chess_parallel():
  os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
  game_name = "dark_chess"
  game_params = tuple()
  test_iters = 2000
  max_trajectory = 100
  rnad_config = rnad.RNaDConfig(
      game_name = game_name, 
      game_params = game_params,
      trajectory_max =  max_trajectory,
      policy_network_layers = [1024, 1024],
      mvs_network_layers = [1024, 1024],
      transformation_network_layers = [1024, 1024],
      
      batch_size = 64,
      learning_rate = 3e-4,
      entropy_schedule_repeats = [100, 1],
      entropy_schedule_size = [500, 2000],
      c_vtrace = 1.0,
      rho_vtrace = 1.0,
      eta_reward_transform = 0.2,

      num_transformations = 10,
      matrix_valued_states = True,
      seed= 4444,
      state_representation=rnad.StateRepresentation.OBSERVATION
  )
  solver = rnad.RNaDSolver(rnad_config)
  print(jax.devices('cpu'))
  # print(jax.devices('gpu'))
  
  
  start = time.time()
  solver.parallel_steps(test_iters)
  print(time.time() - start)
  
  
  start = time.time()
  for i in range(test_iters):
    solver.step()
  print(time.time() - start)
    
    
    
if __name__ == "__main__":
  dark_chess_parallel()