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
import chex
import jax
import jax.numpy as jnp
import numpy as np
import functools 

from collections import namedtuple
 
from open_spiel.python.jax.cfr.jax_cfr import JaxCFR, update_regrets_plus, regret_matching, JAX_CFR_SIMULTANEOUS_UPDATE


@chex.dataclass(frozen=True)
class SePoTCFRConstants:
  players: int
  max_depth: int
  max_actions: int # This includes chance outcomes!  
  transformations: int
  reached_depth_limit: bool
  best_response: bool

  multi_valued_states_ids: int

  max_iset_depth: chex.ArrayTree = () # Is just a list of integers
  isets: chex.ArrayTree = () # Is just a list of integers

  init_reaches: chex.Array = () # For each history in root
  init_iset_reaches: chex.Array = () # For each iset of resolving player
  init_chance: chex.ArrayTree = ()

  multi_valued_states_isets: chex.Array = ()  # For Opponent
  multi_valued_states_actions: chex.Array = () # For opponent
  multi_valued_states_previous_actions: chex.Array = () # For resolving player
  multi_valued_states_utilities: chex.Array = () # From First player perspective
  multi_valued_states_chance_probabilities: chex.Array = () # 

  depth_history_utility: chex.ArrayTree = ()
  depth_history_iset: chex.ArrayTree = () 
  depth_history_actions: chex.ArrayTree = ()
  depth_history_previous_iset: chex.ArrayTree = ()
  depth_history_previous_action: chex.ArrayTree = () 

  depth_history_next_history: chex.ArrayTree = ()
  depth_history_player: chex.ArrayTree = ()
  depth_history_chance: chex.ArrayTree = ()
  depth_history_previous_history: chex.ArrayTree = ()
  depth_history_action_mask: chex.ArrayTree = ()
  depth_history_chance_probabilities: chex.ArrayTree = ()
  
  iset_previous_action: chex.ArrayTree = ()
  iset_action_mask: chex.ArrayTree = ()
  iset_action_depth: chex.ArrayTree = ()
  


class SePoTCFR(JaxCFR):
  def __init__(self, sepot, states: list[pyspiel.State], counterfactual_values: list[float], player_reaches: list[float], chance_reaches:list[float], player:int, depth_limit: int, construct_gadget: bool, multi_valued_states: list = []):
    assert len(states) == len(counterfactual_values)
    assert len(states) == len(player_reaches)
    assert len(states) == len(chance_reaches)
  
    self.timestep = 1
    self.sepot = sepot
    self._linear_averaging = True
    self._regret_matching_plus = True
    self._alternating_updates = True
    self._use_rnad_multi_valued_states = len(multi_valued_states) == 0
    self._predefined_multi_valued_states = multi_valued_states
  
    self.update_regrets = jax.vmap(update_regrets_plus, 0, 0)
    self.game = sepot.rnad._game
    # We add 1 to the depth limit, because we add a gadget 
    depth_limit += int(construct_gadget)

    self.init(states, counterfactual_values, player_reaches, chance_reaches, player, depth_limit, construct_gadget)



  def init(self, states: list[pyspiel.State], counterfactual_values: list[float], player_reaches: list[float], chance_reaches:list[float], player: int, depth_limit: int, construct_gadget: bool):
    opponent = 1 - player
    players = 2
    transformations = self.sepot.rnad.config.num_transformations + 1

    depth_history_utility = [[] for _ in range(players)]
    depth_history_previous_iset = [[] for _ in range(players)]
    depth_history_previous_action = [[] for _ in range(players)]
    depth_history_iset = [[] for _ in range(players)] 
    depth_history_actions = [[] for _ in range(players)]
    depth_history_next_history = []
    depth_history_player = []
    depth_history_chance = []
    depth_history_previous_history = []
    depth_history_action_mask = []
    depth_history_chance_probabilities = [] 

    multi_valued_states_isets = []
    multi_valued_states_actions = []
    multi_valued_states_previous_actions = []
    multi_valued_states_utilities = []
    multi_valued_states_chance_probabilities = []

    # Previous action is mapping of both iset and action!
    iset_previous_action = [[] for _ in range(players)]
    iset_action_mask = [[] for _ in range(players)]
    iset_action_depth = [[] for _ in range(players)]
    ids = [0 for _ in range(players)]
    pl_isets = [{} for _ in range(players)]
    multi_valued_states_ids = [0]
    multi_valued_isets_dict = {}
    distinct_actions = max(self.game.num_distinct_actions(), self.game.max_chance_outcomes())

    # We add 2 layers for gadget. First for the decision, second for terminal/original state
    if construct_gadget:
      for i in range(2):
        for pl in range(players):
          depth_history_utility[pl].append([])
          depth_history_previous_iset[pl].append([])
          depth_history_previous_action[pl].append([])
          depth_history_iset[pl].append([])
          depth_history_actions[pl].append([])
        depth_history_next_history.append([])
        depth_history_player.append([])
        depth_history_chance.append([])
        depth_history_previous_history.append([])
        depth_history_action_mask.append([])
        depth_history_chance_probabilities.append([])

    # Artificial Isets for illegal stuff
    for pl in range(players):
      pl_isets[pl][""] = ids[pl]
      ids[pl] += 1
      am = [0] * distinct_actions
      am[0] = 1
      iset_action_mask[pl].append(am)
      iset_previous_action[pl].append(0)
      iset_action_depth[pl].append(0)

    # ID 0 is reserved for terminal states
    multi_valued_states_ids[0] +=1
    # multi_valued_states_actions.append([0] * transformations)
    # multi_valued_states_utilities.append([0] * transformations)
    # multi_valued_states_previous_actions.append(0)
    # multi_valued_states_isets.append(0)
    # multi_valued_states_chance_probabilities.append(1.0)

    PreviousInfo = namedtuple('PreviousInfo', ('actions', 'isets', 'prev_actions', 'history', 'player'))
        
    def _traverse_tree(state, previous_info, depth, chance = 1.0):

      if depth >= depth_limit:
        if len(depth_history_utility[0]) <= depth:
          for pl in range(players):
            depth_history_utility[pl].append([])
        if state.is_chance_node():
          assert False
        elif state.is_terminal():
          multi_valued_states_isets.append(0)
          multi_valued_states_actions.append([0] * transformations)
          multi_valued_states_utilities.append([0] * transformations)
          multi_valued_states_previous_actions.append(0)
          multi_valued_states_chance_probabilities.append(chance)
          for pl in range(players):
            depth_history_utility[pl][depth].append(state.rewards()[pl])
        else:
        # TODO: Cache states and then call network for batch
          # assert False
          if state.information_state_string(opponent) not in multi_valued_isets_dict:
            multi_valued_isets_dict[state.information_state_string(opponent)] = multi_valued_states_ids[0]
            multi_valued_states_ids[0] += 1
          multi_valued_states = self.sepot.rnad.get_multi_valued_states(state, player)
          # if state.information_state_string(opponent) == "T=4 Player=1 /v_0_0/shot_P0_0_0:H/shot_P1_1_0:H":
          #   print(state.state_tensor())
          multi_valued_states_isets.append(multi_valued_isets_dict[state.information_state_string(opponent)])
          multi_valued_states_actions.append([i + multi_valued_isets_dict[state.information_state_string(opponent)] * transformations for i in range(transformations)])
          multi_valued_states_previous_actions.append(previous_info.actions[player])
          multi_valued_states_chance_probabilities.append(chance)
          if self._use_rnad_multi_valued_states:
            multi_valued_states_utilities.append(multi_valued_states)
          for pl in range(players):
            depth_history_utility[pl][depth].append(0.0)
          
        return
      

      if len(depth_history_next_history) <= depth:
        for pl in range(players):
          depth_history_utility[pl].append([])
          depth_history_previous_iset[pl].append([])
          depth_history_previous_action[pl].append([])
          depth_history_iset[pl].append([])
          depth_history_actions[pl].append([])

        depth_history_next_history.append([])
        depth_history_player.append([])
        depth_history_chance.append([])
        depth_history_previous_history.append([])
        depth_history_action_mask.append([])
        depth_history_chance_probabilities.append([])
        

      history_id = len(depth_history_previous_history[depth])

      next_history_temp = [0] * distinct_actions
      depth_history_next_history[depth].append(next_history_temp)
      depth_history_player[depth].append(state.current_player())
      depth_history_chance[depth].append(chance)
      depth_history_previous_history[depth].append(previous_info.history)

      actions_mask = [0] * distinct_actions
      for a in state.legal_actions():
        actions_mask[a] = 1
      depth_history_action_mask[depth].append(actions_mask)
      chance_probabilities = [0.0 for _ in range(distinct_actions)]
      if state.is_chance_node():
        for a, prob in state.chance_outcomes():
          chance_probabilities[a] = prob
      elif not state.is_terminal():
        chance_probabilities = [1.0 for _ in range(distinct_actions)]
      else:
        chance_probabilities = [1.0/distinct_actions for _ in range(distinct_actions)]
      
      depth_history_chance_probabilities[depth].append(chance_probabilities)
      for pl in range(players):
        depth_history_utility[pl][depth].append(state.rewards()[pl] if not state.is_chance_node() else 0.0)
        depth_history_previous_iset[pl][depth].append(previous_info.isets[pl])
        depth_history_previous_action[pl][depth].append(previous_info.actions[pl])
        if state.current_player() == pl:
          iset = state.information_state_string()
          if iset not in pl_isets[pl]:
            pl_isets[pl][iset] = ids[pl]
            ids[pl] += 1
            iset_previous_action[pl].append(previous_info.actions[pl])
            iset_action_mask[pl].append(actions_mask)
            iset_action_depth[pl].append(previous_info.prev_actions[pl])
          depth_history_iset[pl][depth].append(pl_isets[pl][iset])
          depth_history_actions[pl][depth].append([i + pl_isets[pl][iset] * distinct_actions for i in range(distinct_actions)])
        else:
          depth_history_iset[pl][depth].append(0)
          depth_history_actions[pl][depth].append([0 for _ in range(distinct_actions)])
      # We have reached the depth_limit with gadget. Multi-valued states are trained withing sepot only for decision nodes
      
      for a in state.legal_actions():
        new_chance = chance * chance_probabilities[a]
        assert new_chance > 0.0
        new_actions = tuple(previous_info.actions[pl] if state.current_player() != pl else pl_isets[pl][iset] * distinct_actions + a for pl in range(players))
        new_infosets = tuple(previous_info.isets[pl] if state.current_player() != pl else pl_isets[pl][iset] for pl in range(players))
        new_prev_actions = tuple(previous_info.prev_actions[pl] + int(state.current_player() == pl) for pl in range(players))
        new_info = PreviousInfo(
          new_actions,
          new_infosets,
          new_prev_actions,
          history_id,
          state.current_player(),
          )
        new_state = state.clone()
        new_state.apply_action(a)
      
        # simple workaround if the next element was not visited yet
        next_history_temp[a] = len(depth_history_utility[0][depth + 1]) if len(depth_history_utility[0]) > depth + 1 else 0
        
        _traverse_tree(new_state, new_info, depth + 1, new_chance)
            
    
    if construct_gadget:
      gadget_mask = [0] * distinct_actions
      gadget_mask[0] = 1
      gadget_mask[1] = 1
      for i, state in enumerate(states):
        assert state.current_player() == player
        prev_iset = [0 for _ in range(players)]
        prev_action = [0 for _ in range(players)]

        chance_reach = chance_reaches[i]# * player_reaches[i]

        for pl in range(players):
          depth_history_utility[pl][0].append(0)
          depth_history_previous_iset[pl][0].append(0)
          depth_history_previous_action[pl][0].append(0)
          if pl ==  1 - player:
            iset = "Gadget:" + state.information_state_string(pl)
            if iset not in pl_isets[pl]:
              pl_isets[pl][iset] = ids[pl]
              ids[pl] += 1
              iset_previous_action[pl].append(0)
              
              iset_action_mask[pl].append(gadget_mask)
              iset_action_depth[pl].append(0)
            depth_history_iset[pl][0].append(pl_isets[pl][iset])
            depth_history_actions[pl][0].append([i + pl_isets[pl][iset] * distinct_actions for i in range(distinct_actions)])

            prev_iset[pl] = pl_isets[pl][iset]
            prev_action[pl] = pl_isets[pl][iset] * distinct_actions + 1

          else:
            depth_history_iset[pl][0].append(0)
            depth_history_actions[pl][0].append([0 for _ in range(distinct_actions)])

            prev_iset[pl] = 0
            prev_action[pl] = 0


          depth_history_previous_iset[pl][1].append(prev_iset[pl])
          depth_history_previous_action[pl][1].append(prev_action[pl] - 1 if pl == 1- player else prev_action[pl]) # First action in infoset is the terminal one 
          # Counterfactual values are from the persepective of the player 0.
          depth_history_utility[pl][1].append((1 - (2* pl)) * counterfactual_values[i])
          depth_history_iset[pl][1].append(0)
          depth_history_actions[pl][1].append([0 for _ in range(distinct_actions)])
        

        next_history = [0 for _ in range(distinct_actions)]

        # First is terminal, second is the original state
        next_history[0] = len(depth_history_player[1])
        next_history[1] = len(depth_history_player[1]) + 1 

        depth_history_next_history[0].append(next_history)
        depth_history_player[0].append(1 - player)
        depth_history_chance[0].append(chance_reach)
        depth_history_previous_history[0].append(0)
        depth_history_action_mask[0].append(gadget_mask)
        depth_history_chance_probabilities[0].append([1.0 for _ in range(distinct_actions)]) # Should be chance * policy of player

        depth_history_next_history[1].append([0 for _ in range(distinct_actions)])
        depth_history_player[1].append(-4) # Terminal
        depth_history_chance[1].append(chance_reach)
        depth_history_previous_history[1].append(i)
        depth_history_action_mask[1].append([0 for _ in range(distinct_actions)])
        depth_history_chance_probabilities[1].append([1.0/distinct_actions for _ in range(distinct_actions)])

        prev_amount_actions = [ 0 for _ in range(players)]
        prev_amount_actions[1 - player] = 1

        _traverse_tree(state, PreviousInfo(tuple(prev_action), tuple(prev_iset), tuple(prev_amount_actions), i, 1 - player), 1, chance_reach) 
        
    else:
      # assert False # TODO: Just for testing, remove this later
      for i, state in enumerate(states):
        _traverse_tree(state, PreviousInfo((0, 0), (0, 0), (0, 0), i, 1 - player), 0, chance_reaches[i])

    init_iset_reaches = np.ones(ids[player])
    for state, reach in zip(states, player_reaches):
      assert state.current_player() == player
      iset = state.information_state_string()
      if init_iset_reaches[pl_isets[player][iset]] < 1.0:
        assert init_iset_reaches[pl_isets[player][iset]]  == reach
      init_iset_reaches[pl_isets[player][iset]] = reach

    def convert_to_jax(x):
      return [jnp.asarray(i) for i in x]
    
    def convert_to_jax_players(x):
      return [[jnp.asarray(i) for i in x[pl]] for pl in range(players)]


    depth_history_utility = convert_to_jax_players(depth_history_utility)
    depth_history_iset = convert_to_jax_players(depth_history_iset)
    depth_history_previous_iset = convert_to_jax_players(depth_history_previous_iset)
    depth_history_actions = convert_to_jax_players(depth_history_actions)
    depth_history_previous_action = convert_to_jax_players(depth_history_previous_action)


    depth_history_next_history = convert_to_jax(depth_history_next_history)
    depth_history_player = convert_to_jax(depth_history_player)
    depth_history_chance = convert_to_jax(depth_history_chance)
    depth_history_previous_history = convert_to_jax(depth_history_previous_history)
    depth_history_chance_probabilities = convert_to_jax(depth_history_chance_probabilities)
    depth_history_action_mask = convert_to_jax(depth_history_action_mask)

    max_iset_depth = [np.max(iset_action_depth[pl]) for pl in range(players)]
    iset_previous_action = convert_to_jax(iset_previous_action)
    iset_action_mask = convert_to_jax(iset_action_mask)
    iset_action_depth = convert_to_jax(iset_action_depth) 
    
    if not self._use_rnad_multi_valued_states:
      multi_valued_states_utilities = self._predefined_multi_valued_states

    if multi_valued_states_ids[0] == 1:
      multi_valued_states_utilities = []
      multi_valued_states_isets = []
      multi_valued_states_actions = []
      multi_valued_states_previous_actions = []
      multi_valued_states_chance_probabilities = []

    self.constants = SePoTCFRConstants(
      players = players,
      max_depth = int(len(depth_history_utility[0])),
      max_actions = distinct_actions,
      resolving_player = player,
      transformations = transformations,
      reached_depth_limit = len(multi_valued_states_actions) > 0,
      best_response = True,

      max_iset_depth = max_iset_depth,
      isets = ids,
      multi_valued_states_ids = multi_valued_states_ids[0],

      init_reaches = jnp.asarray(player_reaches),
      # init_reaches = jnp.ones_like(jnp.asarray(player_reaches)),
      init_iset_reaches = jnp.asarray(init_iset_reaches),
      init_chance = jnp.asarray(chance_reaches),

      multi_valued_states_isets = jnp.asarray(multi_valued_states_isets),
      multi_valued_states_actions = jnp.asarray(multi_valued_states_actions),
      multi_valued_states_previous_actions = jnp.asarray(multi_valued_states_previous_actions),
      multi_valued_states_utilities = jnp.asarray(multi_valued_states_utilities),
      multi_valued_states_chance_probabilities = jnp.asarray(multi_valued_states_chance_probabilities),

      # depth_acting_players = jnp.asarray(depth_acting_players),

      depth_history_utility = depth_history_utility,
      depth_history_iset = depth_history_iset,
      depth_history_actions = depth_history_actions,
      depth_history_previous_iset = depth_history_previous_iset,
      depth_history_previous_action = depth_history_previous_action,

      
      depth_history_next_history = depth_history_next_history,
      depth_history_player = depth_history_player,
      depth_history_chance = depth_history_chance,
      depth_history_previous_history = depth_history_previous_history,
      depth_history_action_mask = depth_history_action_mask,
      depth_history_chance_probabilities = depth_history_chance_probabilities,

      iset_previous_action = iset_previous_action,
      iset_action_mask = iset_action_mask,
      iset_action_depth = iset_action_depth,
    )
    self.regrets = [jnp.zeros((ids[pl], distinct_actions)) for pl in range(players)]
    self.averages = [jnp.zeros((ids[pl], distinct_actions)) for pl in range(players)]
    
    self.regret_matching = jax.vmap(regret_matching, 0, 0)
    self.iset_map = pl_isets
    self.mvs_iset_map = multi_valued_isets_dict


  @functools.partial(jax.jit, static_argnums=(0,))
  def jit_step(self, regrets, averages, average_policy_update_coefficient, player):
    """Performs the CFR step.

    This consists of:
    1. Computes the current strategies based on regrets
    2. Computes the realization plan for each action from top of the tree down
    3. Compute the counterfactual regrets from bottom of the tree up
    4. Updates regrets and average stretegies
    Args:
      regrets: Cummulative regrets for all players, list[Float[Isets, Actions]]
      averages: Average strategies for all players, list[Float[Isets, Actions]]
      average_policy_update_coefficient: Weight of the average policy update. When enabled linear_averging it is equal to current iteration. Otherwise 1, int
      player: Player for which the update should be done. When alternating updates are distables, it is JAX_CFR_SIMULTANEOUS_UPDATE
    """ 

    current_strategies = [self.regret_matching(regrets[pl], self.constants.iset_action_mask[pl]) for pl in range(self.constants.players)]

    weighted_strategies = [jnp.copy(current_strategies[pl]) for pl in range(self.constants.players)]
    weighted_strategies[self.constants.resolving_player] = weighted_strategies[self.constants.resolving_player] * self.constants.init_iset_reaches[..., jnp.newaxis]

    realization_plans = self.propagate_strategy(weighted_strategies)
    iset_reaches = [jnp.sum(realization_plans[pl], -1) for pl in range(self.constants.players)]
    # In last row, there are only terminal, so we start row before it
    if self.constants.reached_depth_limit:
      weighted_multi_valued_states = self.constants.multi_valued_states_utilities * (realization_plans[self.constants.resolving_player].ravel()[self.constants.multi_valued_states_previous_actions] * self.constants.multi_valued_states_chance_probabilities)[..., jnp.newaxis]
      bin_regrets = jnp.bincount(self.constants.multi_valued_states_actions.ravel(), weighted_multi_valued_states.ravel(), length = self.constants.multi_valued_states_ids * self.constants.transformations)
      bin_regrets = bin_regrets.reshape(-1, self.constants.transformations)
      # Values are from perspective of resolving player, so opponent chooses minimum
      best_action = jnp.where(self.constants.resolving_player == 0, jnp.argmin(bin_regrets, -1), jnp.argmax(bin_regrets, -1))
      best_action_per_state = best_action[self.constants.multi_valued_states_isets]
      multi_valued_states_utility =jnp.choose(best_action_per_state, self.constants.multi_valued_states_utilities.T, mode='clip')
      
      depth_utils = [[jnp.where(self.constants.multi_valued_states_isets > 0, (1 - (2 *pl)) * multi_valued_states_utility, self.constants.depth_history_utility[pl][-1])] for pl in range(self.constants.players)]
    else:
      depth_utils = [[self.constants.depth_history_utility[pl][-1]] for pl in range(self.constants.players)]
    for i in range(self.constants.max_depth -2, -1, -1):
        
      each_history_policy = self.constants.depth_history_chance_probabilities[i]
      for pl in range(self.constants.players):
        each_history_policy = each_history_policy * jnp.where(self.constants.depth_history_player[i][..., jnp.newaxis] == pl, current_strategies[pl][self.constants.depth_history_iset[pl][i]], 1)
        
      for pl in range(self.constants.players):
        action_value = jnp.where(self.constants.depth_history_player[i][..., jnp.newaxis] == -4, self.constants.depth_history_utility[pl][i][..., jnp.newaxis], depth_utils[pl][-1][self.constants.depth_history_next_history[i]])
        history_value = jnp.sum(action_value * each_history_policy, -1)
        regret = (action_value - history_value[..., jnp.newaxis]) * self.constants.depth_history_action_mask[i] * (self.constants.depth_history_player[i][..., jnp.newaxis] == pl) * self.constants.depth_history_chance[i][..., jnp.newaxis]
        regret = regret * realization_plans[1 - pl].ravel()[self.constants.depth_history_previous_action[1 - pl][i]][..., jnp.newaxis]
        # This is here to avoid artificial infosets of resolving player in root.
        if i == 0:
          regret = regret * self.constants.init_reaches[..., jnp.newaxis]
        bin_regrets = jnp.bincount(self.constants.depth_history_actions[pl][i].ravel(), regret.ravel(), length = self.constants.isets[pl] * self.constants.max_actions)
        bin_regrets = bin_regrets.reshape(-1, self.constants.max_actions)
        regrets[pl] = jnp.where(jnp.logical_or(player == pl, player == JAX_CFR_SIMULTANEOUS_UPDATE), regrets[pl] + bin_regrets, regrets[pl])
        depth_utils[pl].append(history_value) 

    regrets = [self.update_regrets(regrets[pl]) for pl in range(self.constants.players)]
 
    averages = [jnp.where(jnp.logical_or(player == pl, player == JAX_CFR_SIMULTANEOUS_UPDATE), averages[pl] + current_strategies[pl]  * iset_reaches[pl][..., jnp.newaxis]  * average_policy_update_coefficient, averages[pl]) for pl in range(self.constants.players)]

    return regrets, averages 
  
  @functools.partial(jax.jit, static_argnums=(0,))
  def jit_step_stateful(self, regrets, averages, mvs_regrets, average_policy_update_coefficient, player):
    """Performs the CFR step.

    This consists of:
    1. Computes the current strategies based on regrets
    2. Computes the realization plan for each action from top of the tree down
    3. Compute the counterfactual regrets from bottom of the tree up
    4. Updates regrets and average stretegies
    Args:
      regrets: Cummulative regrets for all players, list[Float[Isets, Actions]]
      averages: Average strategies for all players, list[Float[Isets, Actions]]
      average_policy_update_coefficient: Weight of the average policy update. When enabled linear_averging it is equal to current iteration. Otherwise 1, int
      player: Player for which the update should be done. When alternating updates are distables, it is JAX_CFR_SIMULTANEOUS_UPDATE
    """ 

    current_strategies = [self.regret_matching(regrets[pl], self.constants.iset_action_mask[pl]) for pl in range(self.constants.players)]

    weighted_strategies = [jnp.copy(current_strategies[pl]) for pl in range(self.constants.players)]
    weighted_strategies[self.constants.resolving_player] = weighted_strategies[self.constants.resolving_player] * self.constants.init_iset_reaches[..., jnp.newaxis]

    realization_plans = self.propagate_strategy(weighted_strategies)
    iset_reaches = [jnp.sum(realization_plans[pl], -1) for pl in range(self.constants.players)]
    # In last row, there are only terminal, so we start row before it
    if self.constants.reached_depth_limit:
      mvs_current = self.regret_matching(mvs_regrets, jnp.ones_like(mvs_regrets))
      
      mvs_curr_policy = mvs_current[self.constants.multi_valued_states_isets]
      action_value = self.constants.multi_valued_states_utilities * (2 * self.constants.resolving_player - 1) # we are interested in opponents_value
      history_value = jnp.sum(action_value * mvs_curr_policy, -1)
      regret = (action_value - history_value[..., jnp.newaxis]) 
      regret = regret * (realization_plans[self.constants.resolving_player].ravel()[self.constants.multi_valued_states_previous_actions] * self.constants.multi_valued_states_chance_probabilities)[..., jnp.newaxis]

      bin_regrets = jnp.bincount(self.constants.multi_valued_states_actions.ravel(), regret.ravel(), length = self.constants.multi_valued_states_ids * self.constants.transformations)
      bin_regrets = bin_regrets.reshape(-1, self.constants.transformations)
      mvs_regrets = jnp.where(jnp.logical_or(player == 1 - self.constants.resolving_player, player == JAX_CFR_SIMULTANEOUS_UPDATE), mvs_regrets + bin_regrets, mvs_regrets)
      mvs_regrets = jnp.maximum(mvs_regrets, 0.0)
      depth_utils = [[-history_value], [history_value]]


      # weighted_multi_valued_states = self.constants.multi_valued_states_utilities * (realization_plans[self.constants.resolving_player].ravel()[self.constants.multi_valued_states_previous_actions] * self.constants.multi_valued_states_chance_probabilities)[..., jnp.newaxis]
      # bin_regrets = jnp.bincount(self.constants.multi_valued_states_actions.ravel(), weighted_multi_valued_states.ravel(), length = self.constants.multi_valued_states_ids * self.constants.transformations)
      # bin_regrets = bin_regrets.reshape(-1, self.constants.transformations)
      # # Values are from perspective of resolving player, so opponent chooses minimum
      # best_action = jnp.where(self.constants.resolving_player == 0, jnp.argmin(bin_regrets, -1), jnp.argmax(bin_regrets, -1))
      # best_action_per_state = best_action[self.constants.multi_valued_states_isets]
      # multi_valued_states_utility =jnp.choose(best_action_per_state, self.constants.multi_valued_states_utilities.T, mode='clip')
      
      # depth_utils = [[jnp.where(self.constants.multi_valued_states_isets > 0, (1 - (2 *pl)) * multi_valued_states_utility, self.constants.depth_history_utility[pl][-1])] for pl in range(self.constants.players)]
    else:
      depth_utils = [[self.constants.depth_history_utility[pl][-1]] for pl in range(self.constants.players)]
    for i in range(self.constants.max_depth -2, -1, -1):
        
      each_history_policy = self.constants.depth_history_chance_probabilities[i]
      for pl in range(self.constants.players):
        each_history_policy = each_history_policy * jnp.where(self.constants.depth_history_player[i][..., jnp.newaxis] == pl, current_strategies[pl][self.constants.depth_history_iset[pl][i]], 1)
        
      for pl in range(self.constants.players):
        action_value = jnp.where(self.constants.depth_history_player[i][..., jnp.newaxis] == -4, self.constants.depth_history_utility[pl][i][..., jnp.newaxis], depth_utils[pl][-1][self.constants.depth_history_next_history[i]])
        history_value = jnp.sum(action_value * each_history_policy, -1)
        regret = (action_value - history_value[..., jnp.newaxis]) * self.constants.depth_history_action_mask[i] * (self.constants.depth_history_player[i][..., jnp.newaxis] == pl) * self.constants.depth_history_chance[i][..., jnp.newaxis]
        regret = regret * realization_plans[1 - pl].ravel()[self.constants.depth_history_previous_action[1 - pl][i]][..., jnp.newaxis]
        # This is here to avoid artificial infosets of resolving player in root.
        if i == 0:
          regret = regret * self.constants.init_reaches[..., jnp.newaxis]
        bin_regrets = jnp.bincount(self.constants.depth_history_actions[pl][i].ravel(), regret.ravel(), length = self.constants.isets[pl] * self.constants.max_actions)
        bin_regrets = bin_regrets.reshape(-1, self.constants.max_actions)
        regrets[pl] = jnp.where(jnp.logical_or(player == pl, player == JAX_CFR_SIMULTANEOUS_UPDATE), regrets[pl] + bin_regrets, regrets[pl])
        depth_utils[pl].append(history_value) 

    regrets = [self.update_regrets(regrets[pl]) for pl in range(self.constants.players)]
 
    averages = [jnp.where(jnp.logical_or(player == pl, player == JAX_CFR_SIMULTANEOUS_UPDATE), averages[pl] + current_strategies[pl]  * iset_reaches[pl][..., jnp.newaxis]  * average_policy_update_coefficient, averages[pl]) for pl in range(self.constants.players)]

    return regrets, averages, mvs_regrets
  
  def average_policy_dict(self, player: int = -1):
    """Extracts the average_policy from the JAX structures into the TabularPolicy""" 
    averages = [np.asarray(self.averages[pl]) for pl in range(self.constants.players)]
    averages = [averages[pl] / np.sum(averages[pl], -1, keepdims=True) for pl in range(self.constants.players)]

    avg_strategy = {}

    for pl in range(2):
      if player >= 0 and player != pl:
        continue
      for iset, val in self.iset_map[pl].items():
        if iset == '' or iset.startswith('Gadget:'):
          continue
        avg_strategy[iset] = averages[pl][val]
    return avg_strategy
  
  def multiple_steps_stateful(self, iterations):
    self.mvs_regrets = jnp.zeros((self.constants.multi_valued_states_ids, self.constants.transformations))
    for i in range(iterations):
      averaging_coefficient = i if self._linear_averaging else 1
      if self._alternating_updates:
        for player in range(self.constants.players):
          self.regrets, self.averages, self.mvs_regrets = self.jit_step_stateful(self.regrets, self.averages, self.mvs_regrets, averaging_coefficient, player)
        
      else:
        self.regrets, self.averages, self.mvs_regrets = self.jit_step_stateful(self.regrets, self.averages, self.mvs_regrets, averaging_coefficient, JAX_CFR_SIMULTANEOUS_UPDATE)