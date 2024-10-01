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

#TODO(kubicon) ensure generation of infosets works for both players! (even if it does not make sense when opponent acts)

def reconstruct_full_game_test(game):
  iset_neighbor_mapping = {}
  iset_string_to_tensor = {} # TODO: You should test this in a different test with each game!
  iset_tensor_to_string = {}
  iset_to_state = {}
  iset_to_player = {}
  
  def traverse_tree_for_iset(state):
    if state.is_terminal():
      return
    if not state.is_chance_node():
      p1_iset_string = state.information_state_string(0)
      p2_iset_string = state.information_state_string(1)
      p1_iset = ",".join([str(i) for i in state.information_state_tensor(0)])
      p2_iset = ",".join([str(i) for i in state.information_state_tensor(1)])
      if p1_iset_string not in iset_neighbor_mapping:
        assert p1_iset_string not in iset_string_to_tensor
        assert p1_iset not in iset_tensor_to_string
        assert p1_iset_string not in iset_to_state
        assert p1_iset_string not in iset_to_player
        iset_neighbor_mapping[p1_iset_string] = set()
        iset_to_state[p1_iset_string] = set()
        iset_string_to_tensor[p1_iset_string] = p1_iset
        iset_tensor_to_string[p1_iset] = p1_iset_string
        iset_to_player[p1_iset_string] = 0
        
      if p2_iset_string not in iset_neighbor_mapping:
        assert p2_iset_string not in iset_string_to_tensor
        assert p2_iset not in iset_tensor_to_string
        assert p2_iset_string not in iset_to_state
        assert p2_iset_string not in iset_to_player
        iset_neighbor_mapping[p2_iset_string] = set()
        iset_to_state[p2_iset_string] = set()
        iset_string_to_tensor[p2_iset_string] = p2_iset
        iset_tensor_to_string[p2_iset] = p2_iset_string
        iset_to_player[p2_iset_string] = 1
      iset_neighbor_mapping[p1_iset_string].add(p2_iset_string)
      iset_neighbor_mapping[p2_iset_string].add(p1_iset_string)
      iset_to_state[p1_iset_string].add(state)
      iset_to_state[p2_iset_string].add(state)
      
    for a in state.legal_actions():
      new_s = state.clone()
      new_s.apply_action(a)
      traverse_tree_for_iset(new_s)
      
  public_states_isets = []
  def find_public(iset, ps):
    if iset in ps:
      return
    ps.add(iset)
    for neighbor in iset_neighbor_mapping[iset]:
      find_public(neighbor, ps)
    
  
  state = game.new_initial_state()
  traverse_tree_for_iset(state)
  
  for iset in iset_neighbor_mapping.keys():
    call_inner = True
    for ps in public_states_isets :
      if iset in ps:
        call_inner = False
        break
    if call_inner:
      current_ps = set()
      find_public(iset, current_ps)
      public_states_isets.append(current_ps)

  # for ps_isets in public_states_isets:
  #   public_state_states_orig = set()
  #   for iset in ps_isets:
  #     public_state_states_orig = public_state_states_orig.union(iset_to_state[iset])
  #   public_state_states_orig = list(public_state_states_orig)
  #   public_state_states_reconstructed, construct_subgame = reconstruct_public_state.reconstruct_states(public_state_states_orig[0], 0, reconstruct_public_state.ReconstructType.PUBLIC_STATE, 0)
    
  
  #   public_state_states_histories = [s.history() for s in public_state_states_orig]
  #   public_state_states_reconstructed_histories = [s.history() for s in public_state_states_reconstructed]
    
  #   assert construct_subgame == True
  #   assert len(public_state_states_histories) == len(public_state_states_reconstructed_histories)
  #   for s in public_state_states_histories:
  #     assert s in public_state_states_reconstructed_histories
  #   for s in public_state_states_reconstructed_histories:
  #     assert s in public_state_states_histories
      
  for iset, states in iset_to_state.items():
    isets_states = list(states)
    iset_reconstructed, construct_subgame = reconstruct_public_state.reconstruct_states(isets_states[0], 0, reconstruct_public_state.ReconstructType.INFOSET, iset_to_player[iset])
    
    iset_histories = [s.history() for s in isets_states]
    iset_reconstructed_histories = [s.history() for s in iset_reconstructed]
    
    assert construct_subgame == True
    assert len(iset_histories) == len(iset_reconstructed_histories)
    for s in iset_histories:
      assert s in iset_reconstructed_histories
    for s in iset_reconstructed_histories:
      assert s in iset_histories
    
  

def reconstruct_goofspiel_3_descending():
  game_params = {
      "num_cards":3,
      "imp_info": True,
      "points_order": "descending"
      }
  game = pyspiel.load_game_as_turn_based("goofspiel", game_params)
  reconstruct_full_game_test(game)
  
def reconstruct_goofspiel_4_descending():
  game_params = {
      "num_cards":4,
      "imp_info": True,
      "points_order": "descending"
      }
  game = pyspiel.load_game_as_turn_based("goofspiel", game_params)
  reconstruct_full_game_test(game)
  
def reconstruct_goofspiel_3_random():
  game_params = {
      "num_cards":3,
      "imp_info": True,
      "points_order": "random"
      }
  game = pyspiel.load_game_as_turn_based("goofspiel", game_params)
  reconstruct_full_game_test(game)
  
  
def reconstruct_goofspiel_4_random():
  game_params = {
      "num_cards":4,
      "imp_info": True,
      "points_order": "random"
      }
  game = pyspiel.load_game_as_turn_based("goofspiel", game_params)
  reconstruct_full_game_test(game)
  
  
def reconstruct_battleship_2x2_2ships():
  game_params ={"board_height": 2,
                "board_width": 2, 
                "ship_sizes": "[2]", 
                "ship_values": "[1]", 
                "num_shots": 4, 
                "allow_repeated_shots": False}
  game = pyspiel.load_game("battleship", game_params) 
  reconstruct_full_game_test(game)

def reconstruct_goofspiel_init_test():
  game_params = {
      "num_cards":3,
      "imp_info": True,
      "points_order": "descending"
      }
  game = pyspiel.load_game_as_turn_based("goofspiel", game_params)
  state = game.new_initial_state()

  public_state, construct_subgame = reconstruct_public_state.reconstruct(state, 1)
  assert construct_subgame == True
  assert len(public_state) == 1
  assert len(public_state[0]) == 0
  
  infoset, construct_subgame = reconstruct_public_state.reconstruct(state, 1, reconstruct_public_state.ReconstructType.INFOSET, 0)
  
  assert construct_subgame == True
  assert len(infoset) == 1
  assert len(infoset[0]) == 0
  

  
def reconstruct_goofspiel_player_2_test():
  game_params = {
      "num_cards":3,
      "imp_info": True,
      "points_order": "descending"
      }
  game = pyspiel.load_game_as_turn_based("goofspiel", game_params)
  state = game.new_initial_state()

  state.apply_action(0)
  public_state, construct_subgame = reconstruct_public_state.reconstruct(state, 1)
  assert construct_subgame == True
  assert len(public_state) == 3
  for hist in public_state:
    assert len(hist) == 1
    assert hist[0] >= 0
    assert hist[0] < 3
    
  infoset, construct_subgame = reconstruct_public_state.reconstruct(state, 1, reconstruct_public_state.ReconstructType.INFOSET, 1)
  
  assert construct_subgame == True
  assert len(infoset) == 3
  for hist in infoset:
    assert len(hist) == 1
    assert hist[0] >= 0
    assert hist[0] < 3
  


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

  public_state, construct_subgame = reconstruct_public_state.reconstruct(state, 0)
  assert construct_subgame == True
  assert len(public_state) == 3
  for s in public_state:
    assert len(s) == 2
    for i in range(len(s) // 2):
      assert s[i] <= s[i + 1]
      
  infoset, construct_subgame = reconstruct_public_state.reconstruct(state, 0, reconstruct_public_state.ReconstructType.INFOSET, 0)
  
  assert construct_subgame == True
  assert len(infoset) == 2
  for s in infoset:
    assert s[0] == 0
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

  public_state, construct_subgame = reconstruct_public_state.reconstruct(state, 0)
  assert construct_subgame == True
  assert len(public_state) == 3
  for s in public_state:
    assert len(s) == 2
    for i in range(len(s) // 2):
      assert s[i] >= s[i + 1]
      
  infoset, construct_subgame = reconstruct_public_state.reconstruct(state, 0, reconstruct_public_state.ReconstructType.INFOSET, 0)

  assert construct_subgame == True
  assert len(infoset) == 2
  for s in infoset:
    assert s[0] == 2
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
  


  public_state, construct_subgame = reconstruct_public_state.reconstruct(state, 0)
  assert construct_subgame == True
  assert len(public_state) == 1
  for s in public_state:
    assert len(s) == 2
    for i in range(len(s) // 2):
      assert s[i] == 1
      assert s[i] == s[i + 1]
      
  infoset, construct_subgame = reconstruct_public_state.reconstruct(state, 0, reconstruct_public_state.ReconstructType.INFOSET, 0)

  assert construct_subgame == True
  assert len(infoset) == 1
  for s in infoset:
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
  iset_possibles = []
  for i in range(5):
    for j in range(5):
      for k in range(5):
        for l in range(5):
          if i == 0 or j == 0 or k == 0 or l == 0 or i == k or j == l:
            continue
          if i > j and k < l:
            possibles.append([i, j, k, l, 0, 0])
            if i == 3 and k == 1:
              iset_possibles.append([i, j, k, l, 0, 0])
            
  public_state, construct_subgame = reconstruct_public_state.reconstruct(state, 0)
  assert construct_subgame == True
  assert len(public_state) == len(possibles)
  for s in public_state:
    assert len(s) == 6
    assert s in possibles
  for s in possibles:
    assert s in public_state
    
    
  infoset, construct_subgame = reconstruct_public_state.reconstruct(state, 0, reconstruct_public_state.ReconstructType.INFOSET, 0)
  assert construct_subgame == True
  assert len(infoset) == len(iset_possibles)
  for s in infoset:
    assert len(s) == 6
    assert s in iset_possibles
  for s in iset_possibles:
    assert s in infoset

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

  public_state, construct_subgame = reconstruct_public_state.reconstruct(state, 2)
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
  state.apply_action(1)

  public_state, construct_subgame = reconstruct_public_state.reconstruct(state, 0)
  assert construct_subgame == True
  assert len(public_state) == 3
  for s in public_state:
    assert len(s) == 4
    
    for i in range(len(s) // 3):
      assert s[i + 1] <= s[i +2]


def reconstruct_battleship_init_test():
  game_params ={"board_height": 3,
                "board_width": 3, 
                "ship_sizes": "[2;2]", 
                "ship_values": "[1;1]", 
                "num_shots": 9, 
                "allow_repeated_shots": False}
  game = pyspiel.load_game("battleship", game_params) 
  state = game.new_initial_state()
  public_state, construct_subgame = reconstruct_public_state.reconstruct(state, 1)
  assert construct_subgame == True
  assert len(public_state) == 1
  assert len(public_state[0]) == 0
  
  iset, construct_subgame = reconstruct_public_state.reconstruct(state, 1, reconstruct_public_state.ReconstructType.INFOSET, 0)
  assert construct_subgame == True
  assert len(iset) == 1
  assert len(iset[0]) == 0
  
  iset, construct_subgame = reconstruct_public_state.reconstruct(state, 1, reconstruct_public_state.ReconstructType.INFOSET, 1)
  assert construct_subgame == True
  assert len(iset) == 1
  assert len(iset[0]) == 0
  

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
  public_state, construct_subgame = reconstruct_public_state.reconstruct(state, 12)
  assert construct_subgame == True
  assert len(public_state) == 12
  for s in public_state:
    assert len(s) == 1
    assert s[0] in possible_actions
  
  iset, construct_subgame = reconstruct_public_state.reconstruct(state, 0, reconstruct_public_state.ReconstructType.INFOSET, 0)
  assert construct_subgame == True
  assert len(iset) == 1
  assert len(iset[0]) == 1
  assert iset[0][0] == 13
  
  iset, construct_subgame = reconstruct_public_state.reconstruct(state, 0, reconstruct_public_state.ReconstructType.INFOSET, 1)
  assert construct_subgame == True
  assert len(iset) == 12
  for s in iset:
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
  public_state, construct_subgame = reconstruct_public_state.reconstruct(state, 144)
  assert construct_subgame == True
  assert len(public_state) == 144
  for s in public_state:
    assert len(s) == 2
    assert s[0] in possible_actions
    assert s[1] in possible_actions
    
  iset, construct_subgame = reconstruct_public_state.reconstruct(state, 0, reconstruct_public_state.ReconstructType.INFOSET, 0)
  assert construct_subgame == True
  assert len(iset) == 12
  for s in iset:
    assert len(s) == 2
    assert s[0] == 13
    assert s[1] in possible_actions
    
    
  iset, construct_subgame = reconstruct_public_state.reconstruct(state, 0, reconstruct_public_state.ReconstructType.INFOSET, 1)
  assert construct_subgame == True
  assert len(iset) == 12
  for s in iset:
    assert len(s) == 2
    assert s[0] in possible_actions
    assert s[1] == 9
    


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
  public_state, construct_subgame = reconstruct_public_state.reconstruct(state, 12* 88)
  assert construct_subgame == True
  assert len(public_state) == 12 * 88
  for s in public_state:
    assert len(s) == 3
    new_state = game.new_initial_state()
    for a in s:
      new_state.apply_action(a)
      
  iset, construct_subgame = reconstruct_public_state.reconstruct(state, 0, reconstruct_public_state.ReconstructType.INFOSET, 0)
  assert construct_subgame == True
  assert len(iset) == 12
  for s in iset:
    assert len(s) == 3
    assert s[0] == 13
    assert s[2] == 9
    new_state = game.new_initial_state()
    for a in s:
      new_state.apply_action(a)
      
  iset, construct_subgame = reconstruct_public_state.reconstruct(state, 0, reconstruct_public_state.ReconstructType.INFOSET, 1)
  assert construct_subgame == True
  assert len(iset) == 88
  for s in iset:
    assert len(s) == 3
    assert s[1] == 9
    new_state = game.new_initial_state()
    for a in s:
      new_state.apply_action(a)
  
  

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
  public_state, construct_subgame = reconstruct_public_state.reconstruct(state, 88* 88)
  assert construct_subgame == True
  assert len(public_state) == 88 * 88
  for s in public_state:
    assert len(s) == 4
    new_state = game.new_initial_state()
    for a in s:
      new_state.apply_action(a)
      
  iset, construct_subgame = reconstruct_public_state.reconstruct(state, 0, reconstruct_public_state.ReconstructType.INFOSET, 0)
  assert construct_subgame == True
  assert len(iset) == 88
  for s in iset:
    assert len(s) == 4
    assert s[0] == 13
    assert s[2] == 9
    new_state = game.new_initial_state()
    for a in s:
      new_state.apply_action(a)
      
  iset, construct_subgame = reconstruct_public_state.reconstruct(state, 0, reconstruct_public_state.ReconstructType.INFOSET, 1)
  assert construct_subgame == True
  assert len(iset) == 88
  for s in iset:
    assert len(s) == 4
    assert s[1] == 9
    assert s[3] == 21
    new_state = game.new_initial_state()
    for a in s:
      new_state.apply_action(a)

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
  public_state, construct_subgame = reconstruct_public_state.reconstruct(state, 8)
  assert construct_subgame == True
  assert len(public_state) == 8
  for s in public_state:
    assert len(s) == 3
    assert s[2] == 1
    new_state= game.new_initial_state()
    for a in s:
      new_state.apply_action(a)
      
      
  iset, construct_subgame = reconstruct_public_state.reconstruct(state, 0, reconstruct_public_state.ReconstructType.INFOSET, 0)
  assert construct_subgame == True
  assert len(iset) == 2
  for s in iset:
    assert len(s) == 3
    assert s[0] == 4
    assert s[1] == 6 or s[1] == 8
    assert s[2] == 1
    
  iset, construct_subgame = reconstruct_public_state.reconstruct(state, 0, reconstruct_public_state.ReconstructType.INFOSET, 1)
  assert construct_subgame == True
  assert len(iset) == 4
  for s in iset:
    assert len(s) == 3
    assert s[1] == 8
    new_state= game.new_initial_state()
    for a in s:
      new_state.apply_action(a)
      


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
  public_state, construct_subgame = reconstruct_public_state.reconstruct(state, 7)
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
  public_state, construct_subgame = reconstruct_public_state.reconstruct(state, 2)
  assert construct_subgame == True
  assert len(public_state) == 2
  for s in public_state:
    assert len(s) == 5
    assert s[2] == 1
    assert s[3] == 0
    assert s[4] == 3
    new_state= game.new_initial_state()
    for a in s:
      new_state.apply_action(a)
      
      
  iset, construct_subgame = reconstruct_public_state.reconstruct(state, 0, reconstruct_public_state.ReconstructType.INFOSET, 0)
  assert construct_subgame == True
  assert len(iset) == 1
  for s in iset:
    assert len(s) == 5
    assert s[0] == 4
    assert s[1] == 8
    assert s[2] == 1
    assert s[3] == 0
    assert s[4] == 3
    
  iset, construct_subgame = reconstruct_public_state.reconstruct(state, 0, reconstruct_public_state.ReconstructType.INFOSET, 1)
  assert construct_subgame == True
  assert len(iset) == 2
  for s in iset:
    assert len(s) == 5
    assert s[0] == 4 or s[0] == 8
    assert s[1] == 8
    assert s[2] == 1
    assert s[3] == 0
    assert s[4] == 3
    new_state= game.new_initial_state()
    for a in s:
      new_state.apply_action(a)
      

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
  public_state, construct_subgame = reconstruct_public_state.reconstruct(state, 88* 40)
  assert construct_subgame == True
  assert len(public_state) == 88 * 40
  for s in public_state:
    new_state = game.new_initial_state()
    assert len(s) == 5
    for a in s:
      new_state.apply_action(a)
  

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
  public_state, construct_subgame = reconstruct_public_state.reconstruct(state, 36* 5)
  assert construct_subgame == True
  assert len(public_state) == 36 * 5
  for s in public_state:
    new_state = game.new_initial_state()
    assert len(s) == 9
    assert s[4] == 0
    assert s[5] == 8
    assert s[6] == 4
    assert s[7] == 4
    assert s[8] == 1
    for a in s:
      new_state.apply_action(a)
  
def reconstruct_height_battleship():
  game_params ={"board_height": 8,
                "board_width": 1, 
                "ship_sizes": "[4]", 
                "ship_values": "[1]", 
                "num_shots": 8, 
                "allow_repeated_shots": False}
  game = pyspiel.load_game("battleship", game_params) 
  state = game.new_initial_state()
  state.apply_action(16)
  state.apply_action(16)
  public_state, construct_subgame = reconstruct_public_state.reconstruct(state, 3000)
  
  assert construct_subgame == True
  assert len(public_state) == 25
  for s in public_state:
    new_state = game.new_initial_state()
    assert len(s) == 2
    for a in s:
      new_state.apply_action(a)
  
  
def reconstruct_width_battleship():
  game_params ={"board_height": 1,
                "board_width": 8, 
                "ship_sizes": "[4]", 
                "ship_values": "[1]", 
                "num_shots": 8, 
                "allow_repeated_shots": False}
  game = pyspiel.load_game("battleship", game_params) 
  state = game.new_initial_state()
  state.apply_action(8)
  state.apply_action(8)
  public_state, construct_subgame = reconstruct_public_state.reconstruct(state, 3000)
  
  assert construct_subgame == True
  assert len(public_state) == 25
  for s in public_state:
    new_state = game.new_initial_state()
    assert len(s) == 2
    for a in s:
      new_state.apply_action(a)
  
  
  
# TODO: Do more comprehensive test for battleships.
if __name__ == "__main__":
  # reconstruct_goofspiel_3_descending()
  # reconstruct_goofspiel_4_descending()
  # reconstruct_goofspiel_3_random()
  # reconstruct_goofspiel_4_random()
  # reconstruct_battleship_2x2_2ships()
  # reconstruct_goofspiel_init_test()
  # reconstruct_goofspiel_player_2_test()
  # reconstruct_goofspiel_lose_test()
  # reconstruct_goofspiel_win_test()
  # reconstruct_goofspiel_draw_test()
  # reconstruct_goofspiel_larger_test()
  # reconstruct_goofspiel_stop_generating_test()
  # reconstruct_goofspiel_randomized_test()
  # reconstruct_battleship_init_test()
  # reconsturct_battleship_single_ship()
  # reconstruct_battleship_both_ships()
  # reconstruct_battleships_p1_more_ships()
  # reconstruct_battleships_all_ships_placed()
  # reconstruct_battleship_early_stop()
  # reconstruct_small_battleship_more_hits()
  # reconstruct_battleship_hits()
  # reconstruct_battleship_sunken_ship()
  reconstruct_height_battleship()
  reconstruct_width_battleship()