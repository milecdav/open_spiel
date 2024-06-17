from open_spiel.python.algorithms.rnad import rnad as rnad
import pickle 
from open_spiel.python.algorithms.get_all_states import get_all_states
from open_spiel.python import policy
import jax
import jax.numpy as jnp
from open_spiel.python.algorithms import exploitability
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--epsilon", default=0.0, type=float, help="Length of each iteration in seconds")

def compute_exploitability(solver):
  game = solver._game
  rnad_pols = policy.TabularPolicy(game)
  all_states = get_all_states(
      game,
      depth_limit=-1,
      include_terminals=False,
      include_chance_states=False,
      stop_if_encountered=False,
      to_string=lambda s: s.information_state_string())
  rollout = jax.vmap(solver.network.apply, (None, 0), 0)
  
  batch_states = []
  batch_isets = []
  for iset in rnad_pols.state_lookup:
    batch_states.append(solver._state_as_env_step(all_states[iset]))
    batch_isets.append(iset)
  # batch_states = jnp.asarray(batch_states)
  pis, _, _ ,_ = rollout(solver.params_target, jax.tree_util.tree_map(lambda *e: jnp.stack(e, axis=0), *batch_states))
  for i in range(len(batch_isets)):
    state_policy = rnad_pols.policy_for_key(batch_isets[i])
    for a in range(game.num_distinct_actions()):
      state_policy[a] = pis[i][a]
  exp = exploitability.exploitability(game, rnad_pols)
  print(exp, flush=True)

def main():
  args = parser.parse_args([] if "__file__" not in globals() else None)
  cards = 5
  trajectory_max = (cards - 1) * 2
  game = "goofspiel"
  game_params = {"num_cards": cards, "points_order": "descending", "imp_info": True}
  game_params = [(k, v) for k, v in game_params.items()]
  schedule_step = 10000
  config = rnad.RNaDConfig(game_name = game, game_params=game_params, 
                           batch_size=32, 
                           entropy_schedule_repeats = (20, 1 ), 
                           entropy_schedule_size = (schedule_step, schedule_step * 4),
                           learning_rate = 3e-4,
                           c_vtrace = 1.0,
                           trajectory_max = trajectory_max,
                           epsilon=args.epsilon
  )
  solver = rnad.RNaDSolver(config)
  # compute_exploitability(solver)
  for i in range(100):
    compute_exploitability(solver)
    for j in range(schedule_step):
      solver.step()
  # with open("mu_zero_test.pkl", "wb") as f:
  #   pickle.dump(solver, f)


if __name__ == "__main__":
  main()