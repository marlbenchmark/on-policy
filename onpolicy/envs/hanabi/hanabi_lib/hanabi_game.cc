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

#include "hanabi_game.h"

#include "util.h"

namespace hanabi_learning_env {

namespace {
// Constants.
const int kDefaultPlayers = 2;
const int kInformationTokens = 8;
const int kLifeTokens = 3;
const bool kDefaultRandomStart = false;
}  // namespace

HanabiGame::HanabiGame(
    const std::unordered_map<std::string, std::string>& params) {
  params_ = params;
  num_players_ = ParameterValue<int>(params_, "players", kDefaultPlayers);
  REQUIRE(num_players_ >= MinPlayers() && num_players_ <= MaxPlayers());
  num_colors_ = ParameterValue<int>(params_, "colors", kMaxNumColors);
  REQUIRE(num_colors_ > 0 && num_colors_ <= kMaxNumColors);
  num_ranks_ = ParameterValue<int>(params_, "ranks", kMaxNumRanks);
  REQUIRE(num_ranks_ > 0 && num_ranks_ <= kMaxNumRanks);
  hand_size_ = ParameterValue<int>(params_, "hand_size", HandSizeFromRules());
  max_information_tokens_ = ParameterValue<int>(
      params_, "max_information_tokens", kInformationTokens);
  max_life_tokens_ =
      ParameterValue<int>(params_, "max_life_tokens", kLifeTokens);
  seed_ = ParameterValue<int>(params_, "seed", -1);
  random_start_player_ =
      ParameterValue<bool>(params_, "random_start_player", kDefaultRandomStart);
  observation_type_ = AgentObservationType(ParameterValue<int>(
      params_, "observation_type", AgentObservationType::kCardKnowledge));
  while (seed_ == -1) {
    seed_ = std::random_device()();
  }
  rng_.seed(seed_);

  // Work out number of cards per color, and check deck size is large enough.
  cards_per_color_ = 0;
  for (int rank = 0; rank < num_ranks_; ++rank) {
    cards_per_color_ += NumberCardInstances(0, rank);
  }
  REQUIRE(hand_size_ * num_players_ <= cards_per_color_ * num_colors_);

  // Build static list of moves.
  for (int uid = 0; uid < MaxMoves(); ++uid) {
    moves_.push_back(ConstructMove(uid));
  }
  for (int uid = 0; uid < MaxChanceOutcomes(); ++uid) {
    chance_outcomes_.push_back(ConstructChanceOutcome(uid));
  }
}

int HanabiGame::MaxMoves() const {
  return MaxDiscardMoves() + MaxPlayMoves() + MaxRevealColorMoves() +
         MaxRevealRankMoves();
}

int HanabiGame::GetMoveUid(HanabiMove move) const {
  return GetMoveUid(move.MoveType(), move.CardIndex(), move.TargetOffset(),
                    move.Color(), move.Rank());
}

int HanabiGame::GetMoveUid(HanabiMove::Type move_type, int card_index,
                           int target_offset, int color, int rank) const {
  switch (move_type) {
    case HanabiMove::kDiscard:
      return card_index;
    case HanabiMove::kPlay:
      return MaxDiscardMoves() + card_index;
    case HanabiMove::kRevealColor:
      return MaxDiscardMoves() + MaxPlayMoves() +
             (target_offset - 1) * NumColors() + color;
    case HanabiMove::kRevealRank:
      return MaxDiscardMoves() + MaxPlayMoves() + MaxRevealColorMoves() +
             (target_offset - 1) * NumRanks() + rank;
    default:
      return -1;
  }
}

int HanabiGame::MaxChanceOutcomes() const { return NumColors() * NumRanks(); }

int HanabiGame::GetChanceOutcomeUid(HanabiMove move) const {
  if (move.MoveType() != HanabiMove::kDeal) {
    return -1;
  }
  return move.Color() * NumRanks() + move.Rank();
}

HanabiMove HanabiGame::PickRandomChance(
    const std::pair<std::vector<HanabiMove>, std::vector<double>>&
        chance_outcomes) const {
  std::discrete_distribution<std::mt19937::result_type> dist(
      chance_outcomes.second.begin(), chance_outcomes.second.end());
  return chance_outcomes.first[dist(rng_)];
}

std::unordered_map<std::string, std::string> HanabiGame::Parameters() const {
  return {{"players", std::to_string(num_players_)},
          {"colors", std::to_string(NumColors())},
          {"ranks", std::to_string(NumRanks())},
          {"hand_size", std::to_string(HandSize())},
          {"max_information_tokens", std::to_string(MaxInformationTokens())},
          {"max_life_tokens", std::to_string(MaxLifeTokens())},
          {"seed", std::to_string(seed_)},
          {"random_start_player", random_start_player_ ? "true" : "false"},
          {"observation_type", std::to_string(observation_type_)}};
}

int HanabiGame::NumberCardInstances(int color, int rank) const {
  if (color < 0 || color >= NumColors() || rank < 0 || rank >= NumRanks()) {
    return 0;
  }
  if (rank == 0) {
    return 3;
  } else if (rank == NumRanks() - 1) {
    return 1;
  }
  return 2;
}

int HanabiGame::GetSampledStartPlayer() const {
  if (random_start_player_) {
    std::uniform_int_distribution<std::mt19937::result_type> dist(
        0, num_players_ - 1);
    return dist(rng_);
  }
  return 0;
}

int HanabiGame::HandSizeFromRules() const {
  if (num_players_ < 4) {
    return 5;
  }
  return 4;
}

// Uid mapping.  h=hand_size, p=num_players, c=colors, r=ranks
// 0, h-1: discard
// h, 2h-1: play
// 2h, 2h+(p-1)c-1: color hint
// 2h+(p-1)c, 2h+(p-1)c+(p-1)r-1: rank hint
HanabiMove HanabiGame::ConstructMove(int uid) const {
  if (uid < 0 || uid >= MaxMoves()) {
    return HanabiMove(HanabiMove::kInvalid, /*card_index=*/-1,
                      /*target_offset=*/-1, /*color=*/-1, /*rank=*/-1);
  }
  if (uid < MaxDiscardMoves()) {
    return HanabiMove(HanabiMove::kDiscard, /*card_index=*/uid,
                      /*target_offset=*/-1, /*color=*/-1, /*rank=*/-1);
  }
  uid -= MaxDiscardMoves();
  if (uid < MaxPlayMoves()) {
    return HanabiMove(HanabiMove::kPlay, /*card_index=*/uid,
                      /*target_offset=*/-1, /*color=*/-1, /*rank=*/-1);
  }
  uid -= MaxPlayMoves();
  if (uid < MaxRevealColorMoves()) {
    return HanabiMove(HanabiMove::kRevealColor, /*card_index=*/-1,
                      /*target_offset=*/1 + uid / NumColors(),
                      /*color=*/uid % NumColors(), /*rank=*/-1);
  }
  uid -= MaxRevealColorMoves();
  return HanabiMove(HanabiMove::kRevealRank, /*card_index=*/-1,
                    /*target_offset=*/1 + uid / NumRanks(),
                    /*color=*/-1, /*rank=*/uid % NumRanks());
}

HanabiMove HanabiGame::ConstructChanceOutcome(int uid) const {
  if (uid < 0 || uid >= MaxChanceOutcomes()) {
    return HanabiMove(HanabiMove::kInvalid, /*card_index=*/-1,
                      /*target_offset=*/-1, /*color=*/-1, /*rank=*/-1);
  }
  return HanabiMove(HanabiMove::kDeal, /*card_index=*/-1,
                    /*target_offset=*/-1,
                    /*color=*/uid / NumRanks() % NumColors(),
                    /*rank=*/uid % NumRanks());
}

}  // namespace hanabi_learning_env
