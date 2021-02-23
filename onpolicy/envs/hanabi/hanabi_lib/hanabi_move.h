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

#ifndef __HANABI_MOVE_H__
#define __HANABI_MOVE_H__

#include <cstdint>
#include <string>

namespace hanabi_learning_env {

// 5 types of moves:
// "Play" card_index    of card in player hand
// "Discard" card_index    of card in player hand
// "RevealColor" target_offset color    hints to player all cards of color
// "RevealRank" target_offset rank    hints to player all cards of given rank
// NOTE: RevealXYZ target_offset field is an offset from the acting player
// "Deal" color rank    deal card with color and rank
// "Invalid"   move is not valid
class HanabiMove {
  // HanabiMove is small, and intended to be passed by value.
 public:
  enum Type { kInvalid, kPlay, kDiscard, kRevealColor, kRevealRank, kDeal };

  HanabiMove(Type move_type, int8_t card_index, int8_t target_offset,
             int8_t color, int8_t rank)
      : move_type_(move_type),
        card_index_(card_index),
        target_offset_(target_offset),
        color_(color),
        rank_(rank) {}
  // Tests whether two moves are functionally equivalent.
  bool operator==(const HanabiMove& other_move) const;
  std::string ToString() const;

  Type MoveType() const { return move_type_; }
  bool IsValid() const { return move_type_ != kInvalid; }
  int8_t CardIndex() const { return card_index_; }
  int8_t TargetOffset() const { return target_offset_; }
  int8_t Color() const { return color_; }
  int8_t Rank() const { return rank_; }

 private:
  Type move_type_ = kInvalid;
  int8_t card_index_ = -1;
  int8_t target_offset_ = -1;
  int8_t color_ = -1;
  int8_t rank_ = -1;
};

}  // namespace hanabi_learning_env

#endif
