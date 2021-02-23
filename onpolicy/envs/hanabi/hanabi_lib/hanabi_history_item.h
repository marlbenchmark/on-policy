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

#ifndef __HANABI_HISTORY_ITEM_H__
#define __HANABI_HISTORY_ITEM_H__

#include <cstdint>
#include <string>

#include "hanabi_move.h"

namespace hanabi_learning_env {

// A move that has been made within a Hanabi game, along with the side-effects
// of making that move.
struct HanabiHistoryItem {
  explicit HanabiHistoryItem(HanabiMove move_made) : move(move_made) {}
  HanabiHistoryItem(const HanabiHistoryItem& past_move) = default;
  std::string ToString() const;

  // Move that was made.
  HanabiMove move;
  // Index of player who made the move.
  int8_t player = -1;
  // Indicator of whether a Play move was successful.
  bool scored = false;
  // Indicator of whether a Play/Discard move added an information token
  bool information_token = false;
  // Color of card that was played or discarded. Valid if color_ >= 0.
  int8_t color = -1;
  // Rank of card that was played or discarded. Valid if rank_ >= 0.
  int8_t rank = -1;
  // Bitmask indicating whether a card was targeted by a RevealX move.
  // Bit_i=1 if color/rank of card_i matches X in a RevealX move.
  // For example, if cards 0 and 3 had rank 2, a RevealRank 2 move
  // would result in a reveal_bitmask of 9  (2^0+2^3).
  uint8_t reveal_bitmask = 0;
  // Bitmask indicating whether a card was newly revealed by a RevealX move.
  // Bit_i=1 if color/rank of card_i was not known before RevealX move.
  // For example, if cards 1, 2, and 4 had color 'R', and the color of
  // card 1 had previously been revealed to be 'R', a RevealRank 'R' move
  // would result in a newly_revealed_bitmask of 20  (2^2+2^4).
  uint8_t newly_revealed_bitmask = 0;
  // Player that received a card from a Deal move.
  int8_t deal_to_player = -1;
};

}  // namespace hanabi_learning_env

#endif
