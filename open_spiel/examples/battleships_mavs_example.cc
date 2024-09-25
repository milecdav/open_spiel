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

class Node {
    public:
        Node() {}
};

// Example code for using CFR+ to solve Kuhn Poker.
int main(int argc, char** argv) {
  const std::shared_ptr<const open_spiel::Game> game =
        open_spiel::LoadGame("battleship", {{"board_width", open_spiel::GameParameter(2)},
                                {"board_height", open_spiel::GameParameter(2)},
                                {"ship_sizes", open_spiel::GameParameter("[2]")},
                                {"ship_values", open_spiel::GameParameter("[1;1]")},
                                {"num_shots", open_spiel::GameParameter(1000)},
                                {"allow_repeated_shots", open_spiel::GameParameter(false)},
                                {"loss_multiplier", open_spiel::GameParameter(1.0)}});
        std::cout << game->GetType().utility << "\n";
}
