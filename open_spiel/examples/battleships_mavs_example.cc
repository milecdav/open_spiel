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

const int NUM_SAMPLES = 4;
std::mt19937 rng_(1);

enum NodeType {
  PLAYER,
  OPPONENT,
  TERMINAL,
  MAVS
}; 

class Node {
  public:
    Node(NodeType type): type(type) {}
    Node(NodeType type, std::unique_ptr<open_spiel::State> stateToSample): 
      type(type), stateToSample(std::move(stateToSample)) {}
    Node(NodeType type, float value): 
      type(type), value(value) {}

    NodeType GetNodeType() const {
      return type;
    }

    float GetValue() const {
      return value;
    }

    void AddChildWithAction(std::shared_ptr<Node> node, open_spiel::Action action) {
      children.push_back(node);
      actions.push_back(action);
    }

    bool IsTerminal() {
      return type == NodeType::TERMINAL;
    }

    bool IsMavs() {
      return type == NodeType::MAVS;
    }

    std::vector<std::shared_ptr<Node>> children;
    std::vector<open_spiel::Action> actions;
    std::unique_ptr<open_spiel::State> stateToSample;

  private:    
    float value;
    NodeType type;
    
};

class SampleStrategy {
  public:
    virtual open_spiel::ActionsAndProbs GetActionsAndProbs(std::vector<open_spiel::Action> actions, bool placingShips) = 0;
};

class SampleStrategyUniform : public SampleStrategy {
  public:
    open_spiel::ActionsAndProbs GetActionsAndProbs(std::vector<open_spiel::Action> actions, bool placingShips) {
      open_spiel::ActionsAndProbs actionsAndProbs(actions.size());
      for(int iAction = 0; iAction < actions.size(); iAction++) {
        actionsAndProbs[iAction] = std::pair<open_spiel::Action, float>(actions[iAction], 1./actions.size());
      }
      return actionsAndProbs;
    }
};

class SampleStrategyLowToHigh : public SampleStrategy {
  public:
    open_spiel::ActionsAndProbs GetActionsAndProbs(std::vector<open_spiel::Action> actions, bool placingShips) {
      if(placingShips) {
        open_spiel::ActionsAndProbs actionsAndProbs(actions.size());
        for(int iAction = 0; iAction < actions.size(); iAction++) {
          actionsAndProbs[iAction] = std::pair<open_spiel::Action, float>(actions[iAction], 1./actions.size());
        }
        return actionsAndProbs;
      }
      open_spiel::ActionsAndProbs actionsAndProbs(actions.size());
      actionsAndProbs[0] = std::pair<open_spiel::Action, float>(actions[0], 1.);
      for(int iAction = 1; iAction < actions.size(); iAction++) {
        actionsAndProbs[iAction] = std::pair<open_spiel::Action, float>(actions[iAction], 0.);
      }
      return actionsAndProbs;
    }
};

class SampleStrategyNotLow : public SampleStrategy {
  public:
    SampleStrategyNotLow(float low_prob = 0.) : low_prob(low_prob) {}

    open_spiel::ActionsAndProbs GetActionsAndProbs(std::vector<open_spiel::Action> actions, bool placingShips) {
      if(placingShips) {
        open_spiel::ActionsAndProbs actionsAndProbs(actions.size());
        for(int iAction = 0; iAction < actions.size(); iAction++) {
          actionsAndProbs[iAction] = std::pair<open_spiel::Action, float>(actions[iAction], 1./actions.size());
        }
        return actionsAndProbs;
      }
      open_spiel::ActionsAndProbs actionsAndProbs(actions.size());
      if(actions.size() == 1) {
        actionsAndProbs[0] = std::pair<open_spiel::Action, float>(actions[0], 1.);
        return actionsAndProbs;
      }
      actionsAndProbs[0] = std::pair<open_spiel::Action, float>(actions[0], low_prob);
      for(int iAction = 1; iAction < actions.size(); iAction++) {
        actionsAndProbs[iAction] = std::pair<open_spiel::Action, float>(actions[iAction], (1. - low_prob)/(actions.size()-1));
      }
      return actionsAndProbs;
    }

  private:
    float low_prob;
};

class SampleStrategyNotHigh : public SampleStrategy {
  public:
    open_spiel::ActionsAndProbs GetActionsAndProbs(std::vector<open_spiel::Action> actions, bool placingShips) {
      if(placingShips) {
        open_spiel::ActionsAndProbs actionsAndProbs(actions.size());
        for(int iAction = 0; iAction < actions.size(); iAction++) {
          actionsAndProbs[iAction] = std::pair<open_spiel::Action, float>(actions[iAction], 1./actions.size());
        }
        return actionsAndProbs;
      }
      open_spiel::ActionsAndProbs actionsAndProbs(actions.size());
      if(actions.size() == 1) {
        actionsAndProbs[0] = std::pair<open_spiel::Action, float>(actions[0], 1.);
        return actionsAndProbs;
      }
      actionsAndProbs[actions.size() - 1] = std::pair<open_spiel::Action, float>(actions.back(), 0.);
      for(int iAction = 1; iAction < actions.size(); iAction++) {
        actionsAndProbs[iAction] = std::pair<open_spiel::Action, float>(actions[iAction], 1./(actions.size()-1));
      }
      return actionsAndProbs;
    }
};

class SampleStrategyShootEven : public SampleStrategy {
  public:
    open_spiel::ActionsAndProbs GetActionsAndProbs(std::vector<open_spiel::Action> actions, bool placingShips) {
      if(placingShips) {
        open_spiel::ActionsAndProbs actionsAndProbs(actions.size());
        for(int iAction = 0; iAction < actions.size(); iAction++) {
          actionsAndProbs[iAction] = std::pair<open_spiel::Action, float>(actions[iAction], 1./actions.size());
        }
        return actionsAndProbs;
      }
      open_spiel::ActionsAndProbs actionsAndProbs(actions.size());
      int numEven = 0;
      for(int iAction = 1; iAction < actions.size(); iAction++) {
        if(actions[iAction] % 2 == 0) {
          numEven++;
        }
      }
      if(numEven > 0) {
        for(int iAction = 1; iAction < actions.size(); iAction++) {
          if(actions[iAction] % 2 == 0) {
            actionsAndProbs[iAction] = std::pair<open_spiel::Action, float>(actions[iAction], 1./numEven);
          } else {
            actionsAndProbs[iAction] = std::pair<open_spiel::Action, float>(actions[iAction], 0.);
          }
        }      
      } else {
        for(int iAction = 0; iAction < actions.size(); iAction++) {
          actionsAndProbs[iAction] = std::pair<open_spiel::Action, float>(actions[iAction], 1./actions.size());
        }
      }
      return actionsAndProbs;
    }
};

class SampleStrategyShootOdd : public SampleStrategy {
  public:
    open_spiel::ActionsAndProbs GetActionsAndProbs(std::vector<open_spiel::Action> actions, bool placingShips) {
      if(placingShips) {
        open_spiel::ActionsAndProbs actionsAndProbs(actions.size());
        for(int iAction = 0; iAction < actions.size(); iAction++) {
          actionsAndProbs[iAction] = std::pair<open_spiel::Action, float>(actions[iAction], 1./actions.size());
        }
        return actionsAndProbs;
      }
      open_spiel::ActionsAndProbs actionsAndProbs(actions.size());
      int numOdd = 0;
      for(int iAction = 1; iAction < actions.size(); iAction++) {
        if(actions[iAction] % 2 == 1) {
          numOdd++;
        }
      }
      if(numOdd > 0) {
        for(int iAction = 1; iAction < actions.size(); iAction++) {
          if(actions[iAction] % 2 == 1) {
            actionsAndProbs[iAction] = std::pair<open_spiel::Action, float>(actions[iAction], 1./numOdd);
          } else {
            actionsAndProbs[iAction] = std::pair<open_spiel::Action, float>(actions[iAction], 0.);
          }
        }      
      } else {
        for(int iAction = 0; iAction < actions.size(); iAction++) {
          actionsAndProbs[iAction] = std::pair<open_spiel::Action, float>(actions[iAction], 1./actions.size());
        }
      }
      return actionsAndProbs;
    }
};

std::shared_ptr<Node> BuildUntilDepthLimit(std::unique_ptr<open_spiel::State> state, open_spiel::Player player, int depthLimit, int depth) {
  std::shared_ptr<Node> node;  
  // Case when we reached the depth limit
  if(depthLimit == depth) {
    node = std::make_shared<Node>(NodeType::MAVS, std::move(state));
    return std::move(node);
  }
  open_spiel::Player currentPlayer = state->CurrentPlayer();
  // Case when node is terminal
  if(player == open_spiel::kTerminalPlayerId) {
    node = std::make_shared<Node>(NodeType::TERMINAL, state->Rewards()[0]);
    return std::move(node);
  }
  // Case when we have player nodes
  if(currentPlayer == player) {
    node = std::make_shared<Node>(NodeType::PLAYER);
  } else {
    node = std::make_shared<Node>(NodeType::OPPONENT);
  }
  for(open_spiel::Action action : state->LegalActions()) {
    node->AddChildWithAction(BuildUntilDepthLimit(state->Child(action), player, depthLimit, depth+1), action);
  }
  return std::move(node);
}

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

void GoThroughTree(std::shared_ptr<Node> node, int &terminal_count, int &nonterminal_count)  {
  if(node->IsTerminal() or node->IsMavs()) {
    terminal_count++;
    return;
  }
  nonterminal_count++;
  for(std::shared_ptr<Node> child : node->children) {
    GoThroughTree(child, terminal_count, nonterminal_count);
  }
}

float OneSample(std::unique_ptr<open_spiel::State> state, std::shared_ptr<SampleStrategy>& playerStrategy, SampleStrategy& opponentStrategy, open_spiel::Player player) {
  int depth = 0;
  while(true) {
    open_spiel::Player currentPlayer = state->CurrentPlayer();
    if(currentPlayer == open_spiel::kTerminalPlayerId) {
      return state->Rewards()[0];
    }
    bool ship_placement = depth < 2;
    if(currentPlayer == player) {
      state->ApplyAction(open_spiel::SampleAction(playerStrategy->GetActionsAndProbs(state->LegalActions(), ship_placement), rng_).first);
    } else {
      state->ApplyAction(open_spiel::SampleAction(opponentStrategy.GetActionsAndProbs(state->LegalActions(), ship_placement), rng_).first);
    }
    depth++;
  }
}

float SampleValue(std::unique_ptr<open_spiel::State> state, std::shared_ptr<SampleStrategy>& playerStrategy, SampleStrategy& opponentStrategy, open_spiel::Player player) {
  float value = 0;
  for(int iSample = 0; iSample < NUM_SAMPLES; iSample++) {
    value = value + OneSample(state->Clone(), playerStrategy, opponentStrategy, player);
  }
  return value / NUM_SAMPLES;
}

void CrawlTreeAndSample(std::shared_ptr<Node> node, std::vector<std::shared_ptr<SampleStrategy>>& playerPortfolio, SampleStrategy& opponentStrategy, open_spiel::Player player) {
  if(node->IsTerminal()) {
    return;
  }
  if(node->IsMavs()) {
    for(int iStrategy = 0; iStrategy < playerPortfolio.size(); iStrategy++) {
      node->AddChildWithAction(std::make_shared<Node>(NodeType::TERMINAL, SampleValue(node->stateToSample->Clone(), playerPortfolio[iStrategy], opponentStrategy, player)), iStrategy);
    }
    return;
  }
  for(std::shared_ptr<Node> child : node->children) {
    CrawlTreeAndSample(child, playerPortfolio, opponentStrategy, player);
  }
}

float GetBestResponseValue(std::vector<std::shared_ptr<Node>> nodes, std::vector<float> probs, SampleStrategy& opponentStrategy) {
  NodeType type = nodes[0]->GetNodeType();
  if(type == NodeType::PLAYER) {
    float bestValue = -1000;
    for(int iAction = 0; iAction < nodes[0]->actions.size(); iAction++) {
      std::vector<std::shared_ptr<Node>> children;
      for(int iNode = 0; iNode < nodes.size(); iNode++) {
        children[iNode] = nodes[iNode]->children[iAction];
      }
      float actionValue = GetBestResponseValue(children, probs, opponentStrategy);
      if(actionValue > bestValue) {
        bestValue = actionValue;
      }
    }
    return bestValue;
  } else if(type == NodeType::MAVS){
    float bestValue = -1000;
    for(int iAction = 0; iAction < nodes[0]->actions.size(); iAction++) {
      float actionValue = 0;
      for(int iNode = 0; iNode < nodes.size(); iNode++) {
        actionValue = actionValue + nodes[iNode]->children[iAction]->GetValue() * probs[iNode];
      }
      if(actionValue > bestValue) {
        bestValue = actionValue;
      }
    }
    return bestValue;
  } else {
    std::vector<std::shared_ptr<Node>> newNodes;
    std::vector<float> newProbs;
    open_spiel::ActionsAndProbs actionsAndProbs = opponentStrategy.GetActionsAndProbs(nodes[0]->actions, true);
    for(int iAction = 0; iAction < actionsAndProbs.size(); iAction++) {
      newNodes.push_back(nodes[0]->children[iAction]);
      newProbs.push_back(actionsAndProbs[iAction].second*probs[0]);
    }
    return GetBestResponseValue(newNodes, newProbs, opponentStrategy);
  }
}

std::pair<open_spiel::Action, float> GetBestResponse(std::shared_ptr<Node> root, SampleStrategy& opponentStrategy) {
  open_spiel::Action bestAction;
  float bestValue = -1000;
  for(int iAction = 0; iAction < root->actions.size(); iAction++) {
    float actionValue = GetBestResponseValue({root->children[iAction]}, {1.}, opponentStrategy);
    // std::cout << root->actions[iAction] << " " << actionValue << "\n";
    if(actionValue > bestValue) {
      bestAction = root->actions[iAction];
      bestValue = actionValue;
    }
  }
  return {bestAction, bestValue};
}

float PlayGame(std::unique_ptr<open_spiel::State> state, SampleStrategy& playerStrategy, SampleStrategy& opponentStrategy, open_spiel::Player player) {
  int depth = 0;
  while(true) {
    open_spiel::Player currentPlayer = state->CurrentPlayer();
    if(currentPlayer == open_spiel::kTerminalPlayerId) {
      return state->Rewards()[0];
    }
    bool ship_placement = depth < 3;
    if(currentPlayer == player) {
      open_spiel::Action action = open_spiel::SampleAction(playerStrategy.GetActionsAndProbs(state->LegalActions(), ship_placement), rng_).first;
      // std::cout << "Player action: " <<action << " " << state->ActionToString(action) << std::endl;
      state->ApplyAction(action);
    } else {
      open_spiel::Action action = open_spiel::SampleAction(opponentStrategy.GetActionsAndProbs(state->LegalActions(), ship_placement), rng_).first;
      // std::cout << "Opponent action: " << action << " " << state->ActionToString(action) << std::endl;
      state->ApplyAction(action);
    }
    depth++;
  }
}

std::vector<float> PlayMatch(std::unique_ptr<open_spiel::State> state, SampleStrategy& playerStrategy, SampleStrategy& opponentStrategy, int numGames) {
  std::vector<float> results(numGames);
  for(int iGame = 0; iGame < numGames; iGame++) {
    results[iGame] = PlayGame(state->Clone(), playerStrategy, opponentStrategy, open_spiel::Player{0});
  }
  return results;
}

float Variance(std::vector<float>& vec) {
  const size_t sz = vec.size();
  if (sz <= 1) {
    return 0.0;
  }

  // Calculate the mean
  const float mean = std::accumulate(vec.begin(), vec.end(), 0.0) / sz;

  // Now calculate the variance
  auto variance_func = [&mean, &sz](float accumulator, const float& val) {
      return accumulator + ((val - mean)*(val - mean) / (sz - 1));
  };

  return std::accumulate(vec.begin(), vec.end(), 0.0, variance_func);
}

float Mean(std::vector<float>& vec) {
  const size_t sz = vec.size();
  if (sz <= 1) {
    return 0.0;
  }

  // Calculate the mean
  return std::accumulate(vec.begin(), vec.end(), 0.0) / sz;
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

  std::vector<std::shared_ptr<SampleStrategy>> playerStrategies;
  playerStrategies.push_back(std::make_shared<SampleStrategyNotLow>());
  playerStrategies.push_back(std::make_shared<SampleStrategyNotHigh>());
  SampleStrategyUniform uniformStrategy = SampleStrategyUniform();
  SampleStrategyNotLow opponentStrategy = SampleStrategyNotLow(0.05);
  SampleStrategyNotLow realOpponentStrategy = SampleStrategyNotLow();
  std::vector<float> winrates;
  for(int iMatch = 0; iMatch < 100; iMatch++) {
    std::unique_ptr<open_spiel::State> state = game->NewInitialState();
    std::shared_ptr<Node> root = BuildUntilDepthLimit(state->Clone(), open_spiel::Player{0}, 2, 0);
    auto start = std::chrono::high_resolution_clock::now();
    CrawlTreeAndSample(root, playerStrategies, opponentStrategy, open_spiel::Player{0});
    std::pair<open_spiel::Action, float> bestResponse = GetBestResponse(root, opponentStrategy);
    
    if(bestResponse.first == 25 or bestResponse.first == 50) {
      winrates.push_back(1);  
      std::cout << 1;
    } else {
      std::cout << 0;
      winrates.push_back(0);  
    }
    std::cout << std::flush;
    // std::vector<float> results = PlayMatch(state->Child(bestResponse.first), uniformStrategy, realOpponentStrategy, 100);
    // std::cout << "Winrate: " << Mean(results) << " with variance " << Variance(results) << std::endl;
    // winrates.push_back(Mean(results));
  }
  std::cout << "Winrate fluctuation: " << Mean(winrates) << " with variance " << Variance(winrates) << std::endl;
  // Part where I measure time
  // for(int i = 0; i < 10; i++) {
  //   auto start = std::chrono::high_resolution_clock::now();
  //   for(int iSample = 0; iSample < 100000; iSample++) {
  //     GetSample(state->Clone());
  //   }
  //   auto stop = std::chrono::high_resolution_clock::now();
  //   auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  //   std::cout << "Elapsed time: " << duration.count() << "\n";
  // }
}
