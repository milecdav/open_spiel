import argparse
import os
import pyspiel
import time
import pickle
import numpy as np


from open_spiel.python.algorithms.sepot.diffusion import FullDiffusionModel


parser = argparse.ArgumentParser()
# Experiments arguments
parser.add_argument("--experiment_type", default="train", type=str, help="Type of the experiment. Either 'train' or 'sample'")

# Diffusion arguments
parser.add_argument("--noise_steps", default=500, type=int, help="Amount of noise steps")
parser.add_argument("--conditional", default=1, type=int, help="Are states generated only for a conditioned public state")
parser.add_argument("--encoder_decoder", default=1, type=int, help="Is the network trained with encoder-decoder for states")
parser.add_argument("--conditional_dim", default=128, type=int, help="Dimension of the encoding of the conditioned public state")
parser.add_argument("--latent_dim", default=256, type=int, help="Dimension of the encoded state. Only if encoder_decoder is true")
parser.add_argument("--hidden_dim", default=256, type=int, help="Dimension of the diffusion hidden layers")
parser.add_argument("--seed", default=42, type=int, help="Random seed")

# Training arguments
parser.add_argument("--training_sampling", default="traj", type=str, help="Sampling method for training. Either 'all' for all states (only for small games), 'traj' for uniformly generated trajectories or 'traj_p' for trajectories sampled by a random policy")
parser.add_argument("--sampled_trajectories", default=10, type=int, help="Amount of sampled trajectories for samplings 'traj' and 'traj_p'")
parser.add_argument("--training_iterations", default=1000001, type=int, help="Amount of training iterations")
parser.add_argument("--save_each", default=100000, type=int, help="Save the model each that many iterations")

# Sampling arguments
parser.add_argument("--clamp_result", default=0, type=int, help="Clamp the state while generating to [-1, 1]")
parser.add_argument("--sampling_algorithm", default="ddpm", type=str, help="Sampling algorithm. Either 'ddpm' or 'ddim'")
parser.add_argument("--model_path", default="diffusion_models/goofspiel_4_descending/model_ns500_c1_ed0_cd128_ld256_hd256_s42_t100000.pkl", type=str, help="Path to the model, if should be loaded fpr sampling")
parser.add_argument("--samples", default=2, type=int, help="Amount of samples to generate")

# Game specific arguments
parser.add_argument("--cards", default=4, type=int, help="Amount of cards in the game")
parser.add_argument("--points_order", default="descending", type=str, help="Order of the points. Either 'ascending', 'descending' or 'random'")
 
 
from pyinstrument import Profiler

def train():
  args = parser.parse_args([] if "__file__" not in globals() else None)
  
  game_params = {"num_cards": args.cards, "points_order": args.points_order, "imp_info": True}
  model = FullDiffusionModel("goofspiel", game_params,
                             noise_steps=args.noise_steps,
                             conditional=args.conditional == 1,
                             encoder_decoder=args.encoder_decoder == 1,
                             cond_dim=args.conditional_dim,
                             latent_dim=args.latent_dim,
                             hidden_dim=args.hidden_dim,
                             sampled_trajectories=args.sampled_trajectories,
                             training_regime=args.training_sampling,
                             clamp_result=args.clamp_result == 1,
                             seed=args.seed
                             )
  
  path_to_save = f"diffusion_models/goofspiel_{args.cards}_{args.points_order}"
  if not os.path.exists(path_to_save):
    os.makedirs(path_to_save)
  for i in range(args.training_iterations):
    model.training(1)
    if i % args.save_each == 0:
      print("Saving model in iteration", i, flush=True)
      file_name = f"model_ns{args.noise_steps}_c{args.conditional}_ed{args.encoder_decoder}_cd{args.conditional_dim}_ld{args.latent_dim}_hd{args.hidden_dim}_s{args.seed}_t{i}.pkl"
      with open(os.path.join(path_to_save, file_name), "wb") as f:
        pickle.dump(model, f)
        
def sample():
  args = parser.parse_args([] if "__file__" not in globals() else None)
  if not os.path.exists(args.model_path):
    print("Path to model does not exist!")
    return
  with open(args.model_path, "rb") as f:
    model = pickle.load(f)
  model.clamp_result = args.clamp_result == 1
  samples = model.sample(model.game.new_initial_state(), args.samples, args.sampling_algorithm)
  # st = np.asarray(model.game.new_initial_state().state_tensor())
  # st_t = model.transform_from_positive(st)
  # print(st)
  # print(model.encode_decode_method(model.params, st_t)[1])
  for i in samples:
    print((i >0.5).astype(np.float32))
    
def main():
  args = parser.parse_args([] if "__file__" not in globals() else None)
  if args.experiment_type == "train":
    train()
  elif args.experiment_type == "sample":
    sample()
  else:
    print("Invalid experiment type")
    
if __name__ == "__main__":
  main()