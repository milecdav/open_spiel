// Copyright 2024 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <string>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/algorithms/tabular_exploitability.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

#include "chrono"

enum NodeType {
  PLAYER,
  OPPONENT,
  TERMINAL,
  MAVS
}; 

class Node {
  public:
    Node() {}

    NodeType GetNodeType() const {
      return type;
    }

    float GetValue() const {
      return value;
    }

  private:
    std::vector<Node> children;
    std::vector<std::string> actionLabels;
    float value;
    NodeType type;
    
};

int GetSample(std::unique_ptr<open_spiel::State> state) {
  while(!state->IsTerminal()) {
    state->ApplyAction(state->LegalActions()[0]);
  }
  return state->Rewards()[0];
}

void GoThroughTree(std::unique_ptr<open_spiel::State> state, int &terminal_count)  {
  if(state->IsTerminal()) {
    float reward = state->Rewards()[0];
    terminal_count++;
    return;
  }
  for(open_spiel::Action action : state->LegalActions()) {
    GoThroughTree(state->Child(action), terminal_count);
  }
}
// Example code for using CFR+ to solve Kuhn Poker.
int main(int argc, char** argv) {
  int depth_limit = 2;
  int samples = 100;

  const std::shared_ptr<const open_spiel::Game> game =
        open_spiel::LoadGame("battleship", {
                                {"board_width", open_spiel::GameParameter(5)},
                                {"board_height", open_spiel::GameParameter(5)},
                                {"ship_sizes", open_spiel::GameParameter("[2;2]")},
                                {"ship_values", open_spiel::GameParameter("[1;1]")},
                                {"num_shots", open_spiel::GameParameter(25)},
                                {"allow_repeated_shots", open_spiel::GameParameter(false)},
                                {"loss_multiplier", open_spiel::GameParameter(1.0)}});
  std::cout << game->GetType().utility << "\n";
  std::unique_ptr<open_spiel::State> state = game->NewInitialState();

  // Part where I measure time
  for(int i = 0; i < 10; i++) {
    auto start = std::chrono::high_resolution_clock::now();
    for(int iSample = 0; iSample < 100000; iSample++) {
      GetSample(state->Clone());
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "Elapsed time: " << duration.count() << "\n";
  }
}
