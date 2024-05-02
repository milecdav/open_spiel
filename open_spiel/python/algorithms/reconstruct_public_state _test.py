# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pyspiel

from open_spiel.python.algorithms import reconstruct_public_state


def reconstruct_goofspiel_init_test():
  game_params = {
      "num_cards":3,
      "imp_info": True,
      "points_order": "descending"
      }
  game = pyspiel.load_game_as_turn_based("goofspiel", game_params)
  state = game.new_initial_state()

  public_state, construct_subgame = reconstruct_public_state.reconstruct_goofspiel(state, 1)
  assert construct_subgame == True
  assert len(public_state) == 1
  assert len(public_state[0]) == 0
  


def reconstruct_goofspiel_lose_test():
  game_params = {
      "num_cards":3,
      "imp_info": True,
      "points_order": "descending"
      }
  game = pyspiel.load_game_as_turn_based("goofspiel", game_params)
  state = game.new_initial_state()
  state.apply_action(0)
  state.apply_action(1)

  public_state, construct_subgame = reconstruct_public_state.reconstruct_goofspiel(state, 0)
  assert construct_subgame == True
  assert len(public_state) == 3
  for s in public_state:
    assert len(s) == 2
    for i in range(len(s) // 2):
      assert s[i] <= s[i + 1]
  
def reconstruct_goofspiel_win_test():
  game_params = {
      "num_cards":3,
      "imp_info": True,
      "points_order": "descending"
      }
  game = pyspiel.load_game_as_turn_based("goofspiel", game_params)
  state = game.new_initial_state()
  state.apply_action(2)
  state.apply_action(1)

  public_state, construct_subgame = reconstruct_public_state.reconstruct_goofspiel(state, 0)
  assert construct_subgame == True
  assert len(public_state) == 3
  for s in public_state:
    assert len(s) == 2
    for i in range(len(s) // 2):
      assert s[i] >= s[i + 1]
  
def reconstruct_goofspiel_draw_test():
  game_params = {
      "num_cards":3,
      "imp_info": True,
      "points_order": "descending"
      }
  game = pyspiel.load_game_as_turn_based("goofspiel", game_params)
  state = game.new_initial_state()
  state.apply_action(1)
  state.apply_action(1)
  


  public_state, construct_subgame = reconstruct_public_state.reconstruct_goofspiel(state, 0)
  assert construct_subgame == True
  assert len(public_state) == 1
  for s in public_state:
    assert len(s) == 2
    for i in range(len(s) // 2):
      assert s[i] == 1
      assert s[i] == s[i + 1]

def reconstruct_goofspiel_larger_test():
  game_params = {
      "num_cards":5,
      "imp_info": True,
      "points_order": "descending"
      }
  game = pyspiel.load_game_as_turn_based("goofspiel", game_params)
  state = game.new_initial_state()
  state.apply_action(3)
  state.apply_action(2)
  state.apply_action(1)
  state.apply_action(4)
  state.apply_action(0)
  state.apply_action(0)
  possibles = []
  for i in range(5):
    for j in range(5):
      for k in range(5):
        for l in range(5):
          if i == 0 or j == 0 or k == 0 or l == 0 or i == k or j == l:
            continue
          if i > j and k < l:
            possibles.append([i, j, k, l, 0, 0])
            
  public_state, construct_subgame = reconstruct_public_state.reconstruct_goofspiel(state, 0)
  assert construct_subgame == True
  assert len(public_state) == len(possibles)
  for s in public_state:
    assert len(s) == 6
    assert s in possibles
  for s in possibles:
    assert s in public_state

def reconstruct_goofspiel_stop_generating_test():
  game_params = {
      "num_cards":3,
      "imp_info": True,
      "points_order": "descending"
      }
  game = pyspiel.load_game_as_turn_based("goofspiel", game_params)
  state = game.new_initial_state()
  state.apply_action(0)
  state.apply_action(1)

  public_state, construct_subgame = reconstruct_public_state.reconstruct_goofspiel(state, 2)
  assert construct_subgame == False
  assert len(public_state) == 2

def reconstruct_goofspiel_randomized_test():
  game_params = {
      "num_cards":3,
      "imp_info": True,
      "points_order": "random"
      }
  game = pyspiel.load_game_as_turn_based("goofspiel", game_params)
  state = game.new_initial_state()
  state.apply_action(2)
  state.apply_action(0)
  state.apply_action(1)

  public_state, construct_subgame = reconstruct_public_state.reconstruct_goofspiel(state, 0)
  assert construct_subgame == True
  assert len(public_state) == 3
  for s in public_state:
    assert len(s) == 2
    for i in range(len(s) // 2):
      assert s[i] <= s[i + 1]


def reconstruct_battleship_init_test():
  game_params ={"board_height": 3,
                "board_width": 3, 
                "ship_sizes": "[2;2]", 
                "ship_values": "[1;1]", 
                "num_shots": 9, 
                "allow_repeated_shots": False}
  game = pyspiel.load_game("battleship", game_params) 
  state = game.new_initial_state()
  public_state, construct_subgame = reconstruct_public_state.reconstruct_battleship(state, 1)
  assert construct_subgame == True
  assert len(public_state) == 1
  assert len(public_state[0]) == 0

def reconsturct_battleship_single_ship():
  game_params ={"board_height": 3,
                "board_width": 3, 
                "ship_sizes": "[2;2]", 
                "ship_values": "[1;1]", 
                "num_shots": 9, 
                "allow_repeated_shots": False}
  game = pyspiel.load_game("battleship", game_params) 
  state = game.new_initial_state()
  possible_actions = state.legal_actions()
  state.apply_action(13)
  public_state, construct_subgame = reconstruct_public_state.reconstruct_battleship(state, 12)
  assert construct_subgame == True
  assert len(public_state) == 12
  for s in public_state:
    assert len(s) == 1
    assert s[0] in possible_actions

def reconstruct_battleship_both_ships():
  game_params ={"board_height": 3,
                "board_width": 3, 
                "ship_sizes": "[2;2]", 
                "ship_values": "[1;1]", 
                "num_shots": 9, 
                "allow_repeated_shots": False}
  game = pyspiel.load_game("battleship", game_params) 
  state = game.new_initial_state()
  possible_actions = state.legal_actions()
  state.apply_action(13)
  state.apply_action(9)
  public_state, construct_subgame = reconstruct_public_state.reconstruct_battleship(state, 144)
  assert construct_subgame == True
  assert len(public_state) == 144
  for s in public_state:
    assert len(s) == 2
    assert s[0] in possible_actions
    assert s[1] in possible_actions


def reconstruct_battleships_p1_more_ships():
  game_params ={"board_height": 3,
                "board_width": 3, 
                "ship_sizes": "[2;2]", 
                "ship_values": "[1;1]", 
                "num_shots": 9, 
                "allow_repeated_shots": False}
  game = pyspiel.load_game("battleship", game_params) 
  state = game.new_initial_state()
  state.apply_action(13)
  state.apply_action(9)
  state.apply_action(9)
  # state.apply_action(21)
  public_state, construct_subgame = reconstruct_public_state.reconstruct_battleship(state, 12* 88)
  assert construct_subgame == True
  assert len(public_state) == 12 * 88
  for s in public_state:
    assert len(s) == 3
    state = game.new_initial_state()
    for a in s:
      state.apply_action(a)

def reconstruct_battleships_all_ships_placed():
  game_params ={"board_height": 3,
                "board_width": 3, 
                "ship_sizes": "[2;2]", 
                "ship_values": "[1;1]", 
                "num_shots": 9, 
                "allow_repeated_shots": False}
  game = pyspiel.load_game("battleship", game_params) 
  state = game.new_initial_state()
  state.apply_action(13)
  state.apply_action(9)
  state.apply_action(9)
  state.apply_action(21)
  public_state, construct_subgame = reconstruct_public_state.reconstruct_battleship(state, 88* 88)
  assert construct_subgame == True
  assert len(public_state) == 88 * 88
  for s in public_state:
    assert len(s) == 4
    state = game.new_initial_state()
    for a in s:
      state.apply_action(a)

def reconstruct_battleship_single_hit():
  game_params ={"board_height": 2,
                "board_width": 2, 
                "ship_sizes": "[2]", 
                "ship_values": "[1]", 
                "num_shots": 4, 
                "allow_repeated_shots": False}
  game = pyspiel.load_game("battleship", game_params) 
  state = game.new_initial_state()
  state.apply_action(4)
  state.apply_action(8)
  state.apply_action(1)
  public_state, construct_subgame = reconstruct_public_state.reconstruct_battleship(state, 8)
  assert construct_subgame == True
  assert len(public_state) == 8
  for s in public_state:
    assert len(s) == 3
    assert s[2] == 1
    state= game.new_initial_state()
    for a in s:
      state.apply_action(a)

def reconstruct_battleship_early_stop():
  game_params ={"board_height": 2,
                "board_width": 2, 
                "ship_sizes": "[2]", 
                "ship_values": "[1]", 
                "num_shots": 4, 
                "allow_repeated_shots": False}
  game = pyspiel.load_game("battleship", game_params) 
  state = game.new_initial_state()
  state.apply_action(4)
  state.apply_action(8)
  state.apply_action(1)
  public_state, construct_subgame = reconstruct_public_state.reconstruct_battleship(state, 7)
  assert construct_subgame == False

def reconstruct_small_battleship_more_hits():
  game_params ={"board_height": 2,
                "board_width": 2, 
                "ship_sizes": "[2]", 
                "ship_values": "[1]", 
                "num_shots": 4, 
                "allow_repeated_shots": False}
  game = pyspiel.load_game("battleship", game_params) 
  state = game.new_initial_state()
  state.apply_action(4)
  state.apply_action(8)
  state.apply_action(1)
  state.apply_action(0)
  state.apply_action(3)
  public_state, construct_subgame = reconstruct_public_state.reconstruct_battleship(state, 2)
  assert construct_subgame == True
  assert len(public_state) == 2
  for s in public_state:
    assert len(s) == 5
    assert s[2] == 1
    assert s[3] == 0
    assert s[4] == 3
    state= game.new_initial_state()
    for a in s:
      state.apply_action(a)

def reconstruct_battleship_hits():
  game_params ={"board_height": 3,
                "board_width": 3, 
                "ship_sizes": "[2;2]", 
                "ship_values": "[1;1]", 
                "num_shots": 9, 
                "allow_repeated_shots": False}
  game = pyspiel.load_game("battleship", game_params) 
  state = game.new_initial_state()
  state.apply_action(13)
  state.apply_action(9)
  state.apply_action(9)
  state.apply_action(21)
  state.apply_action(4)
  public_state, construct_subgame = reconstruct_public_state.reconstruct_battleship(state, 88* 40)
  assert construct_subgame == True
  assert len(public_state) == 88 * 40
  for s in public_state:
    state = game.new_initial_state()
    assert len(s) == 5
    for a in s:
      state.apply_action(a)
  

def reconstruct_battleship_sunken_ship():
  game_params ={"board_height": 3,
                "board_width": 3, 
                "ship_sizes": "[2;2]", 
                "ship_values": "[1;1]", 
                "num_shots": 9, 
                "allow_repeated_shots": False}
  game = pyspiel.load_game("battleship", game_params) 
  state = game.new_initial_state()
  state.apply_action(13)
  state.apply_action(9)
  state.apply_action(9)
  state.apply_action(21)
  state.apply_action(0)
  state.apply_action(8)
  state.apply_action(4)
  state.apply_action(4)
  state.apply_action(1)
  public_state, construct_subgame = reconstruct_public_state.reconstruct_battleship(state, 36* 5)
  assert construct_subgame == True
  assert len(public_state) == 36 * 5
  for s in public_state:
    state = game.new_initial_state()
    assert len(s) == 9
    assert s[4] == 0
    assert s[5] == 8
    assert s[6] == 4
    assert s[7] == 4
    assert s[8] == 1
    for a in s:
      state.apply_action(a)
  
# TODO: Do more comprehensive test for battleships.
if __name__ == "__main__":
  reconstruct_goofspiel_init_test()
  reconstruct_goofspiel_lose_test()
  reconstruct_goofspiel_win_test()
  reconstruct_goofspiel_draw_test()
  reconstruct_goofspiel_larger_test()
  reconstruct_goofspiel_stop_generating_test()
  reconstruct_goofspiel_randomized_test()
  reconstruct_battleship_init_test()
  reconsturct_battleship_single_ship()
  reconstruct_battleship_both_ships()
  reconstruct_battleships_p1_more_ships()
  reconstruct_battleships_all_ships_placed()
  reconstruct_battleship_early_stop()
  reconstruct_small_battleship_more_hits()
  reconstruct_battleship_hits()
  reconstruct_battleship_sunken_ship()