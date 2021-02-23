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

#ifndef __HANABI_OBSERVATION_H__
#define __HANABI_OBSERVATION_H__

#include <string>
#include <vector>

#include "hanabi_card.h"
#include "hanabi_game.h"
#include "hanabi_hand.h"
#include "hanabi_history_item.h"
#include "hanabi_move.h"
#include "hanabi_state.h"

namespace hanabi_learning_env {

// Agent observation of a HanabiState
class HanabiObservation {
 public:
  HanabiObservation(const HanabiState& state, int observing_player);

  std::string ToString() const;

  // offset of current player from observing player.
  int CurPlayerOffset() const { return cur_player_offset_; }
  // observed hands are in relative order, with index 1 being the
  // first player clock-wise from observing_player. hands[0][] has
  // invalid cards as players don't see their own cards.
  const std::vector<HanabiHand>& Hands() const { return hands_; }
  const std::vector<HanabiHand>& OwnHands() const { return ownhands_; }
  // The element at the back is the most recent discard.
  const std::vector<HanabiCard>& DiscardPile() const { return discard_pile_; }
  const std::vector<int>& Fireworks() const { return fireworks_; }
  int DeckSize() const { return deck_size_; }  // number of remaining cards
  const HanabiGame* ParentGame() const { return parent_game_; }
  // Moves made since observing_player's last action, most recent to oldest
  // (that is, last_moves[0] is the most recent move.)
  // Move targets are relative to observing_player not acting_player.
  // Note that the deal moves are included in this vector.
  const std::vector<HanabiHistoryItem>& LastMoves() const {
    return last_moves_;
  }
  int InformationTokens() const { return information_tokens_; }
  int LifeTokens() const { return life_tokens_; }
  const std::vector<HanabiMove>& LegalMoves() const { return legal_moves_; }

  // returns true if card with color and rank can be played on fireworks pile
  bool CardPlayableOnFireworks(int color, int rank) const;
  bool CardPlayableOnFireworks(HanabiCard card) const {
    return CardPlayableOnFireworks(card.Color(), card.Rank());
  }

 private:
  int cur_player_offset_;  // offset of current_player from observing_player
  std::vector<HanabiHand> hands_;         // observing player is element 0
  std::vector<HanabiHand> ownhands_;         // observing player is element 0
  std::vector<HanabiCard> discard_pile_;  // back is most recent discard
  std::vector<int> fireworks_;
  int deck_size_;
  std::vector<HanabiHistoryItem> last_moves_;
  int information_tokens_;
  int life_tokens_;
  std::vector<HanabiMove> legal_moves_;  // list of legal moves
  const HanabiGame* parent_game_ = nullptr;
};

}  // namespace hanabi_learning_env

#endif
