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

#ifndef __HANABI_STATE_H__
#define __HANABI_STATE_H__

#include <random>
#include <string>
#include <vector>

#include "hanabi_card.h"
#include "hanabi_game.h"
#include "hanabi_hand.h"
#include "hanabi_history_item.h"
#include "hanabi_move.h"

namespace hanabi_learning_env {

constexpr int kChancePlayerId = -1;

class HanabiState {
 public:
  class HanabiDeck {
   public:
    explicit HanabiDeck(const HanabiGame& game);
    // DealCard returns invalid card on failure.
    HanabiCard DealCard(int color, int rank);
    HanabiCard DealCard(std::mt19937* rng);
    int Size() const { return total_count_; }
    bool Empty() const { return total_count_ == 0; }
    int CardCount(int color, int rank) const {
      return card_count_[CardToIndex(color, rank)];
    }

   private:
    int CardToIndex(int color, int rank) const {
      return color * num_ranks_ + rank;
    }
    int IndexToColor(int index) const { return index / num_ranks_; }
    int IndexToRank(int index) const { return index % num_ranks_; }

    // Number of instances in the deck for each card.
    // E.g., if card_count_[CardToIndex(card)] == 2, then there are two
    // instances of card remaining in the deck, available to be dealt out.
    std::vector<int> card_count_;
    int total_count_ = -1;  // Total number of cards available to be dealt out.
    int num_ranks_ = -1;    // From game.NumRanks(), used to map card to index.
  };

  enum EndOfGameType {
    kNotFinished,        // Not the end of game.
    kOutOfLifeTokens,    // Players ran out of life tokens.
    kOutOfCards,         // Players ran out of cards.
    kCompletedFireworks  // All fireworks played.
  };

  // Construct a HanabiState, initialised to the start of the game.
  // If start_player >= 0, the game-provided start player is overridden
  // and the first player after chance is start_player.
  explicit HanabiState(const HanabiGame* parent_game, int start_player = -1);
  // Copy constructor for recursive game traversals using copy + apply-move.
  HanabiState(const HanabiState& state) = default;

  bool MoveIsLegal(HanabiMove move) const;
  void ApplyMove(HanabiMove move);
  // Legal moves for state. Moves point into an unchanging list in parent_game.
  std::vector<HanabiMove> LegalMoves(int player) const;
  // Returns true if card with color and rank can be played on fireworks pile.
  bool CardPlayableOnFireworks(int color, int rank) const;
  bool CardPlayableOnFireworks(HanabiCard card) const {
    return CardPlayableOnFireworks(card.Color(), card.Rank());
  }
  bool ChanceOutcomeIsLegal(HanabiMove move) const { return MoveIsLegal(move); }
  double ChanceOutcomeProb(HanabiMove move) const;
  void ApplyChanceOutcome(HanabiMove move) { ApplyMove(move); }
  void ApplyRandomChance();
  // Get the valid chance moves, and associated probabilities.
  // Guaranteed that moves.size() == probabilities.size().
  std::pair<std::vector<HanabiMove>, std::vector<double>> ChanceOutcomes()
      const;
  EndOfGameType EndOfGameStatus() const;
  bool IsTerminal() const { return EndOfGameStatus() != kNotFinished; }
  int Score() const;
  std::string ToString() const;

  int CurPlayer() const { return cur_player_; }
  int LifeTokens() const { return life_tokens_; }
  int InformationTokens() const { return information_tokens_; }
  const std::vector<HanabiHand>& Hands() const { return hands_; }
  const std::vector<int>& Fireworks() const { return fireworks_; }
  const HanabiGame* ParentGame() const { return parent_game_; }
  const HanabiDeck& Deck() const { return deck_; }
  // Get the discard pile (the element at the back is the most recent discard.)
  const std::vector<HanabiCard>& DiscardPile() const { return discard_pile_; }
  // Sequence of moves from beginning of game. Stored as <move, actor>.
  const std::vector<HanabiHistoryItem>& MoveHistory() const {
    return move_history_;
  }

 private:
  // Add card to table if possible, if not lose a life token.
  // Returns <scored,information_token_added>
  // success is true iff card was successfully added to fireworks.
  // information_token_added is true iff information_tokens increase
  // (i.e., success=true, highest rank was added, and not at max tokens.)
  std::pair<bool, bool> AddToFireworks(HanabiCard card);
  const HanabiHand& HandByOffset(int offset) const {
    return hands_[(cur_player_ + offset) % hands_.size()];
  }
  HanabiHand* HandByOffset(int offset) {
    return &hands_[(cur_player_ + offset) % hands_.size()];
  }
  void AdvanceToNextPlayer();  // Set cur_player to next player to act.
  bool HintingIsLegal(HanabiMove move) const;
  int PlayerToDeal() const;  // -1 if no player needs a card.
  bool IncrementInformationTokens();
  void DecrementInformationTokens();
  void DecrementLifeTokens();

  const HanabiGame* parent_game_ = nullptr;
  HanabiDeck deck_;
  // Back element of discard_pile_ is most recently discarded card.
  std::vector<HanabiCard> discard_pile_;
  std::vector<HanabiHand> hands_;
  std::vector<HanabiHistoryItem> move_history_;
  int cur_player_ = -1;
  int next_non_chance_player_ = -1;  // Next non-chance player to act.
  int information_tokens_ = -1;
  int life_tokens_ = -1;
  std::vector<int> fireworks_;
  int turns_to_play_ = -1;  // Number of turns to play once deck is empty.
};

}  // namespace hanabi_learning_env

#endif
