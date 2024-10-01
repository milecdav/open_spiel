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


import enum
import pyspiel
import copy
import numpy as np
from ortools.sat.python import cp_model

class ReconstructType(enum.Enum):
  """A enumeration class for children selection in ISMCTS."""
  INFOSET = 0
  PUBLIC_STATE = 1
  

def reconstruct_states(state: pyspiel.State, limit: int = 0, reconstruct_type: ReconstructType = ReconstructType.PUBLIC_STATE, player: int = 0):
  histories, premature_stop = reconstruct(state, limit, reconstruct_type, player)
  return reconstruct_states_from_histories(state.get_game().new_initial_state(), histories), premature_stop

# Player is important only when ReconstrudctType is INFOSET
def reconstruct(state: pyspiel.State, limit: int = 0, reconstruct_type: ReconstructType = ReconstructType.PUBLIC_STATE, player: int = 0):
  # Could both be turn_based or simultaneous
  assert player >= 0 or reconstruct_type == ReconstructType.PUBLIC_STATE
  
  if "Goofspiel" in state.get_game().get_type().long_name:
    return reconstruct_goofspiel(state, limit, reconstruct_type, player)
  elif state.get_game().get_type().short_name == "battleship":
    return reconstruct_battleship(state, limit, reconstruct_type, player)
  else:
    raise NotImplementedError("Reconstruction for this game is not implemented")

def reconstruct_goofspiel_model(turn_outcomes: list[int], player_bets: list[int], num_cards: int, limit, reconstruct_type: ReconstructType, player: int, points_order: list[int] = []):
  # You always reconstruct after chance node!
  assert len(points_order) == 0 or len(points_order) == len(turn_outcomes) + 1
  model = cp_model.CpModel()
  game_rounds = len(turn_outcomes)
  limit = limit if limit > 0 else float("inf")
  histories = []

  played_cards = [[model.NewIntVar(0, num_cards - 1, name=f"{pl}:{game_round}") for game_round in range(game_rounds)] for pl in range(2)]
  model.AddAllDifferent(played_cards[0])
  model.AddAllDifferent(played_cards[1])

  for i, outcome in enumerate(turn_outcomes):
    # We are only reconstructing infoset so we force the actions of a player
    if reconstruct_type == ReconstructType.INFOSET:
      model.Add(played_cards[player][i] == player_bets[i])
      
    # Tie, both played same card
    if outcome == 0:
      model.Add(played_cards[0][i] == played_cards[1][i])
      model.Add(played_cards[0][i] == player_bets[i]) # TODO(kubicon): Duplicate constraint. Does it matter?
    elif outcome == 1:
      model.Add(played_cards[0][i] > played_cards[1][i])
    elif outcome == -1:
      model.Add(played_cards[0][i] < played_cards[1][i])

  class GoofspielSolutionLimitCallback(cp_model.CpSolverSolutionCallback):

    def __init__(self):
      super().__init__()
      self.__solution_count = 0
      self.__solution_limit = limit
      self.premature_stop_search = False

    def on_solution_callback(self):
      self.__solution_count += 1
      if self.__solution_count > self.__solution_limit:
        self.StopSearch()
        self.premature_stop_search = True
        return
      history = []
      for game_round in range(game_rounds):
        if points_order:
          history.append(points_order[game_round])
        history.append(self.Value(played_cards[0][game_round]))
        history.append(self.Value(played_cards[1][game_round]))
      if points_order:
        history.append(points_order[-1])
      if reconstruct_type == ReconstructType.INFOSET and game_rounds < len(player_bets):
        history.append(player_bets[-1])
      histories.append(history)

  solver = cp_model.CpSolver()

  solver.parameters.enumerate_all_solutions = True
  callback = GoofspielSolutionLimitCallback()
  status = solver.Solve(model, callback)
  # if state
  assert len(histories) <= limit
  return histories, callback.premature_stop_search
  

def reconstruct_goofspiel(state: pyspiel.State, limit: int, reconstruct_type: ReconstructType, player: int):#
  # TOOD: What about turn-based goofspiel
  # assert state.is_chance_node() == False
  assert "Goofspiel" in state.get_game().get_type().long_name
  assert state.current_player() >= 0
  turn_outcomes = []
  player_bets = []
  points_order = []
  num_cards = state.get_game().num_distinct_actions()
  prev_action = -1
  # full_h = state.full_history()
  for h in state.full_history():
    if h.player == 0:
      prev_action = h.action
    elif h.player == 1:
      turn_outcomes.append(1 if h.action < prev_action else -1 if h.action > prev_action else 0)
    elif h.player == -1: # Should be chances
      points_order.append(h.action)
    if h.player == player:
      player_bets.append(h.action)
      
  assert len(points_order) >= (state.get_game().get_parameters()["game"]["points_order"] == "random") # If random, then there has to be atleast one chance node played. If not then, then it has to be zero
  histories, premature_stop = reconstruct_goofspiel_model(turn_outcomes, player_bets, num_cards, limit, reconstruct_type, player, points_order)
  
  # TODO(kubicon) this should be completely removed from creating gadget in the code, so remove this.
  create_cadget = not premature_stop
  if "Turn-based" not in state.get_game().get_type().long_name or state.current_player() == 0 or (reconstruct_type == ReconstructType.INFOSET and player == 0):
    return histories, create_cadget
  tb_histories = []
  randomized = int(state.get_game().get_parameters()["game"]["points_order"] == "random")
  for history in histories:
    for a in range(state.get_game().num_distinct_actions()):
      if a not in history[randomized::2 + randomized]:
        new_history = copy.copy(history)
        new_history.append(a)
        tb_histories.append(new_history)
  return tb_histories, create_cadget

def reconstruct_states_from_histories(init_state, histories):
  states = []
  for h in histories:
    state = init_state.clone()
    for a in h:
      state.apply_action(a)
    states.append(state)
  return states


# TODO(kubicon) does not work when ship is longer than row/column
def reconstruct_battleship(state: pyspiel.State, limit: int, reconstruct_type: ReconstructType, player: int):
  assert state.get_game().get_type().short_name == "battleship"
  game = state.get_game()
  game_params = game.get_parameters()
  board_height = game_params["board_height"]
  board_width = game_params["board_width"]
  ship_sizes = game_params["ship_sizes"][1:-1] # Removes parantheses
  ship_sizes = [int(s) for s in ship_sizes.split(";")]
  full_history = state.full_history()
  ship_positions = np.zeros((2, board_height, board_width), dtype=np.int8)
  # 0 is no shot, 1 is miss, 2 is hit unknown ship, 3 + i is hit of ship i
  shots = np.zeros((2, board_height, board_width), dtype=np.int8)
  # Not shots of a player, but board positions
  shot_results = [[], []]
  positioned_ships = [0, 0]
  hits = np.zeros((2, len(ship_sizes)))
  # (y, x, shot_result)
  # shots = []
  # from [0, board_size), the actions are shots, from [boards_size, 2 * board_size) the actions are horizontal ship placements from [2 * board_size, 3 * board_size) the actions are vertical ship placements
  for i, h in enumerate(full_history):
    # Ship placement
    if i < len(ship_sizes) * 2:
      # P0 placement
      assert h.action >= board_height * board_width
      positioned_ships[h.player] += 1
      placement_action = h.action - board_height * board_width
      ship_orientation = placement_action // (board_height * board_width)
      ship_x = h.action % board_width
      ship_y = h.action % (board_width * board_height) // board_width
      ship_size = ship_sizes[i // 2]
      # Horizontal
      if ship_orientation == 0:
        ship_positions[h.player, ship_y, ship_x:ship_x + ship_size] = i // 2 + 1
      # Vertical
      else:
        ship_positions[h.player, ship_y:ship_y + ship_size, ship_x] = i // 2 + 1
    # Shots
    else:
      shot_x = h.action % board_width
      shot_y = h.action // board_width
      if ship_positions[1 - h.player, shot_y, shot_x] == 0:
        shots[1 - h.player, shot_y, shot_x] = 1
      else:
        ship_id = ship_positions[1 - h.player, shot_y, shot_x] - 1
        shots[1 - h.player, shot_y, shot_x] = ship_id + 3
        hits[1 - h.player, ship_id] += 1
  for i in range(len(ship_sizes)):
    if hits[0, i] != ship_sizes[i]:
      shots[0, shots[0, :, :] == i + 3] = 2
    if hits[1, i] != ship_sizes[i]:
      shots[1, shots[1, :, :] == i + 3] = 2
  for pl, y, x in zip(*np.nonzero(shots)):
    shot_result = shots[pl, y, x]
    shot_results[pl] = (y * x, shot_result)
  # print(hits)
  # Just reconstruct all possible states for n actions
    
  p1_placed_ships = ship_sizes[:(len(full_history) + 1) // 2]
  p2_placed_ships = ship_sizes[:len(full_history) // 2]
  if reconstruct_type == ReconstructType.INFOSET:
    # When finding iset for P1 you go for opponents board 
    if player == 1:
      p1_histories, construct_subgame = reconstruct_positions_battleship(board_height, board_width, p1_placed_ships, shots[0], limit)
      p2_histories = [[h.action for i, h in enumerate(state.full_history()) if h.player == 1 and i < len(ship_sizes) * 2]]
    else:
      p1_histories = [[h.action for i,h in enumerate(state.full_history()) if h.player == 0 and i < len(ship_sizes) * 2]]
      p2_histories, construct_subgame = reconstruct_positions_battleship(board_height, board_width, p2_placed_ships, shots[1], limit)
  else:
    p1_histories, construct_subgame = reconstruct_positions_battleship(board_height, board_width, p1_placed_ships, shots[0])
    if construct_subgame == False:
      return [], False
    p2_histories, construct_subgame = reconstruct_positions_battleship(board_height, board_width, p2_placed_ships, shots[1])
    if construct_subgame == False or len(p1_histories) * len(p2_histories) > limit:
      return [], False
  # print(p1_histories)
  # print(p2_histories)
  histories = []
  for p1_history in p1_histories:
    for p2_history in p2_histories:
      assert len(p1_history) == len(p2_history) or len(p1_history) == len(p2_history) + 1
      new_history = []
      for i in range(len(p2_history)):
        new_history.append(p1_history[i])
        new_history.append(p2_history[i])
      if len(p1_history) > len(p2_history):
        new_history.append(p1_history[-1])
      for h in full_history[2*len(ship_sizes):]:
        new_history.append(h.action)
      histories.append(new_history)
  return histories, True
      






def reconstruct_positions_battleship(board_height, board_width, ships, shots, limit: int = 0):
  model = cp_model.CpModel()
  limit = limit if limit > 0 else float("inf")
  histories = []
  ship_positions = [model.NewIntVar(0, board_height * board_width - 1, name=f"ship_{i}_pos") for i in range(np.sum(ships, dtype=int))]
  # Orientation false is horizontal, true is vertical
  ship_orientations = [model.NewBoolVar(name=f"ship_{i}_orientation") for i in range(len(ships))]
  # max_size_ship = np.max(ships)
  start_indices_ship = np.cumsum([0] + ships)
  for ship_id, ship in enumerate(ships):
    if ship == 1:
      model.add(ship_orientations[ship_id] == 0)
  model.AddAllDifferent(ship_positions)
  known_ship_hits = [0 for _ in ships]
  sunken_ships = {}
  np_iterator = np.nditer(shots, flags=['multi_index'])
  for shot_result in np_iterator:
    # No info
    if shot_result == 0:
      continue
    y, x =  np_iterator.multi_index
    # Miss
    if shot_result == 1:
      for ship_position in ship_positions:
        model.Add(ship_position != y * board_width + x)
    # Hit known ship
    # We start with first index (we go from upper left corner) and say that this is at the position given by model.
    if shot_result > 2:
      ship_id = shot_result - 3
      model.Add(ship_positions[start_indices_ship[ship_id] + known_ship_hits[ship_id]] == y * board_width + x)
      assert known_ship_hits[ship_id] < ships[ship_id]
      known_ship_hits[ship_id] += 1
    # Hit unknown ship
    if shot_result == 2:
      slack_hits = [model.NewBoolVar(name=f"ship_{i}_slack_hit") for i in range(len(ship_positions))]
      model.AddBoolOr(slack_hits)
      # We check how many unknown hits or no shots are right and down from this position
      for ship_pos, slack in zip(ship_positions, slack_hits):
        model.Add(ship_pos == y * board_width + x).OnlyEnforceIf(slack)
      possible_horizontal_hits = 0
      possible_vertical_hits = 0
      for curr_x in range(x, board_width):
        if shots[y, curr_x] != 2 and shots[y, curr_x] != 0:
          break
        possible_horizontal_hits += 1
      for curr_y in range(y, board_height):
        if shots[curr_y, x] != 2 and shots[curr_y, x] != 0:
          break
        possible_vertical_hits += 1
      for ship_id, ship in enumerate(ships):
        if ship > possible_horizontal_hits:
          model.Add(ship_positions[start_indices_ship[ship_id]] != y * board_width + x).OnlyEnforceIf(ship_orientations[ship_id].Not())
        if ship > possible_vertical_hits:
          model.Add(ship_positions[start_indices_ship[ship_id]] != y * board_width + x).OnlyEnforceIf(ship_orientations[ship_id])
  for ship_id, ship in enumerate(ships):
    init_ship_index = start_indices_ship[ship_id]
    y = init_ship_index // board_width
    x = init_ship_index % board_width
    # for next_horizontal_part
    for next_ship_part in range(1, ship):
      
      model.Add(ship_positions[init_ship_index + next_ship_part] == ship_positions[init_ship_index] + next_ship_part).OnlyEnforceIf(ship_orientations[ship_id].Not())
      model.Add(ship_positions[init_ship_index + next_ship_part] == ship_positions[init_ship_index] + next_ship_part * board_width).OnlyEnforceIf(ship_orientations[ship_id])
      # Invalidates ships that would go to the next row
      for row in range(board_height):
        model.Add(ship_positions[init_ship_index] != row * board_width - (next_ship_part)).OnlyEnforceIf(ship_orientations[ship_id].Not())
  
  
  class BattleshipSolutionLimitCallback(cp_model.CpSolverSolutionCallback):

    def __init__(self):
      super().__init__()
      self.__solution_count = 0
      self.__solution_limit = limit
      self.premature_stop_search = False

    def on_solution_callback(self):
      self.__solution_count += 1
      if self.__solution_count > self.__solution_limit:
        self.StopSearch()
        self.premature_stop_search = True
        return
      
      history = []
      for ship_id, ship_orientation in enumerate(ship_orientations):
        vertical = self.Value(ship_orientation)
        initial_position = self.Value(ship_positions[start_indices_ship[ship_id]])
        action = board_width * board_height * (1 + vertical) + initial_position
        assert action >= board_height * board_width
        assert action < 3 * board_height * board_width
        history.append(action) 
      histories.append(history)

  solver = cp_model.CpSolver()

  solver.parameters.enumerate_all_solutions = True
  callback = BattleshipSolutionLimitCallback()
  status = solver.Solve(model, callback)

  assert len(histories) <= limit
  return histories, not callback.premature_stop_search


if __name__ == "__main__":
  # game = pyspiel.load_game("goospiel", {"num_cards": 4, "imp_info": True, "points_order": "descending"})
  # reconstruct_goofspiel(game.new_initial_state(), 100)
  game = pyspiel.load_game("battleship", {"board_height": 3, "board_width": 3, "ship_sizes": "[2;2]", "ship_values": "[1;2]", "num_shots": 4, "allow_repeated_shots": False}) 
  state = game.new_initial_state()
  state.apply_action(13)
  state.apply_action(23)
  state.apply_action(21)
  state.apply_action(9)
  state.apply_action(0)
  state.apply_action(3)
  state.apply_action(1)
  state.apply_action(4)
  state.apply_action(7)
  state.apply_action(2)
  reconstruct(state)