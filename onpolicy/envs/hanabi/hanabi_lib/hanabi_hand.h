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

#ifndef __HANABI_HAND_H__
#define __HANABI_HAND_H__

#include <cstdint>
#include <string>
#include <vector>

#include "hanabi_card.h"

namespace hanabi_learning_env {

class HanabiHand {
 public:
  class ValueKnowledge {
    // Knowledge about an unknown integer variable in range 0 to value_range-1.
    // Records hints that either reveal the exact value (no longer unknown),
    // or reveal that the variable is not some particular value.
    // For example, ValueKnowledge(3) tracks a variable that can be 0, 1, or 2.
    // Initially, ValueHinted()=false, value()=-1, and ValueCouldBe(v)=true
    // for v=0, 1, and 2.
    // After recording that the value is not 1, we have
    // ValueHinted()=false, value()=-1, and ValueCouldBe(1)=false.
    // After recording that the value is 0, we have
    // ValueHinted()=true, value()=0, and ValueCouldBe(v)=false for v=1, and 2.
   public:
    explicit ValueKnowledge(int value_range);
    int Range() const { return value_plausible_.size(); }
    // Returns true if and only if the exact value was revealed.
    // Does not perform inference to get a known value from not-value hints.
    bool ValueHinted() const { return value_ >= 0; }
    int Value() const { return value_; }  // -1 if value was not hinted.
    // Returns true if we have no hint saying variable is not the given value.
    bool IsPlausible(int value) const { return value_plausible_[value]; }
    // Record a hint that gives the value of the variable.
    void ApplyIsValueHint(int value);
    // Record a hint that the variable does not have the given value.
    void ApplyIsNotValueHint(int value);

   private:
    // Value if hint directly provided the value, or -1 with no direct hint.
    int value_ = -1;
    std::vector<bool> value_plausible_;  // Knowledge from not-value hints.
  };

  class CardKnowledge {
    // Hinted knowledge about color and rank of an initially unknown card.
   public:
    CardKnowledge(int num_colors, int num_ranks);
    // Returns number of possible colors being tracked.
    int NumColors() const { return color_.Range(); }
    // Returns true if and only if the exact color was revealed.
    // Does not perform inference to get a known color from not-color hints.
    bool ColorHinted() const { return color_.ValueHinted(); }
    // Color of card if it was hinted, -1 if not hinted.
    int Color() const { return color_.Value(); }
    // Returns true if we have no hint saying card is not the given color.
    bool ColorPlausible(int color) const { return color_.IsPlausible(color); }
    void ApplyIsColorHint(int color) { color_.ApplyIsValueHint(color); }
    void ApplyIsNotColorHint(int color) { color_.ApplyIsNotValueHint(color); }
    // Returns number of possible ranks being tracked.
    int NumRanks() const { return rank_.Range(); }
    // Returns true if and only if the exact rank was revealed.
    // Does not perform inference to get a known rank from not-rank hints.
    bool RankHinted() const { return rank_.ValueHinted(); }
    // Rank of card if it was hinted, -1 if not hinted.
    int Rank() const { return rank_.Value(); }
    // Returns true if we have no hint saying card is not the given rank.
    bool RankPlausible(int rank) const { return rank_.IsPlausible(rank); }
    void ApplyIsRankHint(int rank) { rank_.ApplyIsValueHint(rank); }
    void ApplyIsNotRankHint(int rank) { rank_.ApplyIsNotValueHint(rank); }
    std::string ToString() const;

   private:
    ValueKnowledge color_;
    ValueKnowledge rank_;
  };

  HanabiHand() {}
  HanabiHand(const HanabiHand& hand)
      : cards_(hand.cards_), card_knowledge_(hand.card_knowledge_) {}
  // Copy hand. Hide cards (set to invalid) if hide_cards is true.
  // Hide card knowledge (set to unknown) if hide_knowledge is true.
  HanabiHand(const HanabiHand& hand, bool hide_cards, bool hide_knowledge);
  // Cards and corresponding card knowledge are always arranged from oldest to
  // newest, with the oldest card or knowledge at index 0.
  const std::vector<HanabiCard>& Cards() const { return cards_; }
  const std::vector<CardKnowledge>& Knowledge() const {
    return card_knowledge_;
  }
  void AddCard(HanabiCard card, const CardKnowledge& initial_knowledge);
  // Remove card_index card from hand. Put in discard_pile if not nullptr
  // (pushes the card to the back of the discard_pile vector).
  void RemoveFromHand(int card_index, std::vector<HanabiCard>* discard_pile);
  // Make cards with the given rank visible.
  // Returns new information bitmask, bit_i set if card_i color was revealed
  // and was previously unknown.
  uint8_t RevealRank(int rank);
  // Make cards with the given color visible.
  // Returns new information bitmask, bit_i set if card_i color was revealed
  // and was previously unknown.
  uint8_t RevealColor(int color);
  std::string ToString() const;

 private:
  // A set of cards and knowledge about them.
  std::vector<HanabiCard> cards_;
  std::vector<CardKnowledge> card_knowledge_;
};

}  // namespace hanabi_learning_env

#endif
