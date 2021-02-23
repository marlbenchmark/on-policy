// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef __HANABI_GAME_H__
#define __HANABI_GAME_H__

#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "hanabi_card.h"
#include "hanabi_move.h"

namespace hanabi_learning_env {

class HanabiGame {
 public:
  // An agent's observation of a state does include all state knowledge.
  // For example, observations never include an agent's own cards.
  // A kMinimal observation is similar to what a human sees, and does not
  // include any memory of past RevalColor/RevealRank hints. A CardKnowledge
  // observation includes per-card knowledge of past hints, as well as simple
  // inferred knowledge of the form "this card is not red, because it was
  // not revealed as red in a past <RevealColor Red> move". A Seer observation
  // shows all cards, including the player's own cards, regardless of what
  // hints have been given.
  enum AgentObservationType { kMinimal = 0, kCardKnowledge = 1, kSeer = 2 };

  explicit HanabiGame(
      const std::unordered_map<std::string, std::string>& params);

  // Number of different player moves.
  int MaxMoves() const;
  // Get a HanabiMove by unique id.
  HanabiMove GetMove(int uid) const { return moves_[uid]; }
  // Get unique id for a move. Returns -1 for invalid move.
  int GetMoveUid(HanabiMove move) const;
  int GetMoveUid(HanabiMove::Type move_type, int card_index, int target_offset,
                 int color, int rank) const;
  // Number of different chance outcomes.
  int MaxChanceOutcomes() const;
  // Get a chance-outcome HanabiMove by unique id.
  HanabiMove GetChanceOutcome(int uid) const { return chance_outcomes_[uid]; }
  // Get unique id for a chance-outcome move. Returns -1 for invalid move.
  int GetChanceOutcomeUid(HanabiMove move) const;
  // Randomly sample a random chance-outcome move from list of moves and
  // associated probability distribution.
  HanabiMove PickRandomChance(
      const std::pair<std::vector<HanabiMove>, std::vector<double>>&
          chance_outcomes) const;

  std::unordered_map<std::string, std::string> Parameters() const;
  int MinPlayers() const { return 2; }
  int MaxPlayers() const { return 5; }
  int MinScore() const { return 0; }
  int MaxScore() const { return num_ranks_ * num_colors_; }
  std::string Name() const { return "Hanabi"; }

  int NumColors() const { return num_colors_; }
  int NumRanks() const { return num_ranks_; }
  int NumPlayers() const { return num_players_; }
  int HandSize() const { return hand_size_; }
  int MaxInformationTokens() const { return max_information_tokens_; }
  int MaxLifeTokens() const { return max_life_tokens_; }
  int CardsPerColor() const { return cards_per_color_; }
  int MaxDeckSize() const { return cards_per_color_ * num_colors_; }
  int NumberCardInstances(int color, int rank) const;
  int NumberCardInstances(HanabiCard card) const {
    return NumberCardInstances(card.Color(), card.Rank());
  }
  AgentObservationType ObservationType() const { return observation_type_; }

  // Get the first player to act. Might be randomly generated at each call.
  int GetSampledStartPlayer() const;

 private:
  // Calculating max moves by move type.
  int MaxDiscardMoves() const { return hand_size_; }
  int MaxPlayMoves() const { return hand_size_; }
  int MaxRevealColorMoves() const { return (num_players_ - 1) * num_colors_; }
  int MaxRevealRankMoves() const { return (num_players_ - 1) * num_ranks_; }

  int HandSizeFromRules() const;
  HanabiMove ConstructMove(int uid) const;
  HanabiMove ConstructChanceOutcome(int uid) const;

  // Table of all possible moves in this game.
  std::vector<HanabiMove> moves_;
  // Table of all possible chance outcomes in this game.
  std::vector<HanabiMove> chance_outcomes_;
  std::unordered_map<std::string, std::string> params_;
  int num_colors_ = -1;
  int num_ranks_ = -1;
  int num_players_ = -1;
  int hand_size_ = -1;
  int max_information_tokens_ = -1;
  int max_life_tokens_ = -1;
  int cards_per_color_ = -1;
  int seed_ = -1;
  bool random_start_player_ = false;
  AgentObservationType observation_type_ = kCardKnowledge;
  mutable std::mt19937 rng_;
};

}  // namespace hanabi_learning_env

#endif
