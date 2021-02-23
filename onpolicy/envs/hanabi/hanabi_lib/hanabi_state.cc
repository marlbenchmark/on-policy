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

#include "hanabi_state.h"

#include <algorithm>
#include <cassert>
#include <numeric>

#include "util.h"

namespace hanabi_learning_env {

namespace {
// Returns bitmask of card indices which match color.
uint8_t HandColorBitmask(const HanabiHand& hand, int color) {
  uint8_t mask = 0;
  const auto& cards = hand.Cards();
  assert(cards.size() <= 8);  // More than 8 cards is not supported.
  for (int i = 0; i < cards.size(); ++i) {
    if (cards[i].Color() == color) {
      mask |= static_cast<uint8_t>(1) << i;
    }
  }
  return mask;
}

// Returns bitmask of card indices which match color.
uint8_t HandRankBitmask(const HanabiHand& hand, int rank) {
  uint8_t mask = 0;
  const auto& cards = hand.Cards();
  assert(cards.size() <= 8);  // More than 8 cards is not supported.
  for (int i = 0; i < cards.size(); ++i) {
    if (cards[i].Rank() == rank) {
      mask |= static_cast<uint8_t>(1) << i;
    }
  }
  return mask;
}
}  // namespace

HanabiState::HanabiDeck::HanabiDeck(const HanabiGame& game)
    : card_count_(game.NumColors() * game.NumRanks(), 0),
      total_count_(0),
      num_ranks_(game.NumRanks()) {
  for (int color = 0; color < game.NumColors(); ++color) {
    for (int rank = 0; rank < game.NumRanks(); ++rank) {
      auto count = game.NumberCardInstances(color, rank);
      card_count_[CardToIndex(color, rank)] = count;
      total_count_ += count;
    }
  }
}

HanabiCard HanabiState::HanabiDeck::DealCard(std::mt19937* rng) {
  if (Empty()) {
    return HanabiCard();
  }
  std::discrete_distribution<std::mt19937::result_type> dist(
      card_count_.begin(), card_count_.end());
  int index = dist(*rng);
  assert(card_count_[index] > 0);
  --card_count_[index];
  --total_count_;
  return HanabiCard(IndexToColor(index), IndexToRank(index));
}

HanabiCard HanabiState::HanabiDeck::DealCard(int color, int rank) {
  int index = CardToIndex(color, rank);
  if (card_count_[index] <= 0) {
    return HanabiCard();
  }
  assert(card_count_[index] > 0);
  --card_count_[index];
  --total_count_;
  return HanabiCard(IndexToColor(index), IndexToRank(index));
}

HanabiState::HanabiState(const HanabiGame* parent_game, int start_player)
    : parent_game_(parent_game),
      deck_(*parent_game),
      hands_(parent_game->NumPlayers()),
      cur_player_(kChancePlayerId),
      next_non_chance_player_(start_player >= 0 &&
                                      start_player < parent_game->NumPlayers()
                                  ? start_player
                                  : parent_game->GetSampledStartPlayer()),
      information_tokens_(parent_game->MaxInformationTokens()),
      life_tokens_(parent_game->MaxLifeTokens()),
      fireworks_(parent_game->NumColors(), 0),
      turns_to_play_(parent_game->NumPlayers()) {}

void HanabiState::AdvanceToNextPlayer() {
  if (!deck_.Empty() && PlayerToDeal() >= 0) {
    cur_player_ = kChancePlayerId;
  } else {
    cur_player_ = next_non_chance_player_;
    next_non_chance_player_ = (cur_player_ + 1) % hands_.size();
  }
}

bool HanabiState::IncrementInformationTokens() {
  if (information_tokens_ < ParentGame()->MaxInformationTokens()) {
    ++information_tokens_;
    return true;
  } else {
    return false;
  }
}

void HanabiState::DecrementInformationTokens() {
  assert(information_tokens_ > 0);
  --information_tokens_;
}

void HanabiState::DecrementLifeTokens() {
  assert(life_tokens_ > 0);
  --life_tokens_;
}

std::pair<bool, bool> HanabiState::AddToFireworks(HanabiCard card) {
  if (CardPlayableOnFireworks(card)) {
    ++fireworks_[card.Color()];
    // Check if player completed a stack.
    if (fireworks_[card.Color()] == ParentGame()->NumRanks()) {
      return {true, IncrementInformationTokens()};
    }
    return {true, false};
  } else {
    DecrementLifeTokens();
    return {false, false};
  }
}

bool HanabiState::HintingIsLegal(HanabiMove move) const {
  if (InformationTokens() <= 0) {
    return false;
  }
  if (move.TargetOffset() < 1 ||
      move.TargetOffset() >= ParentGame()->NumPlayers()) {
    return false;
  }
  return true;
}

int HanabiState::PlayerToDeal() const {
  for (int i = 0; i < hands_.size(); ++i) {
    if (hands_[i].Cards().size() < ParentGame()->HandSize()) {
      return i;
    }
  }
  return -1;
}

bool HanabiState::MoveIsLegal(HanabiMove move) const {
  switch (move.MoveType()) {
    case HanabiMove::kDeal:
      if (cur_player_ != kChancePlayerId) {
        return false;
      }
      if (deck_.CardCount(move.Color(), move.Rank()) == 0) {
        return false;
      }
      break;
    case HanabiMove::kDiscard:
      if (InformationTokens() >= ParentGame()->MaxInformationTokens()) {
        return false;
      }
      if (move.CardIndex() >= hands_[cur_player_].Cards().size()) {
        return false;
      }
      break;
    case HanabiMove::kPlay:
      if (move.CardIndex() >= hands_[cur_player_].Cards().size()) {
        return false;
      }
      break;
    case HanabiMove::kRevealColor: {
      if (!HintingIsLegal(move)) {
        return false;
      }
      const auto& cards = HandByOffset(move.TargetOffset()).Cards();
      if (!std::any_of(cards.begin(), cards.end(),
                       [move](const HanabiCard& card) {
                         return card.Color() == move.Color();
                       })) {
        return false;
      }
      break;
    }
    case HanabiMove::kRevealRank: {
      if (!HintingIsLegal(move)) {
        return false;
      }
      const auto& cards = HandByOffset(move.TargetOffset()).Cards();
      if (!std::any_of(cards.begin(), cards.end(),
                       [move](const HanabiCard& card) {
                         return card.Rank() == move.Rank();
                       })) {
        return false;
      }
      break;
    }
    default:
      return false;
  }
  return true;
}

void HanabiState::ApplyMove(HanabiMove move) {
  REQUIRE(MoveIsLegal(move));
  if (deck_.Empty()) {
    --turns_to_play_;
  }
  HanabiHistoryItem history(move);
  history.player = cur_player_;
  switch (move.MoveType()) {
    case HanabiMove::kDeal: {
        history.deal_to_player = PlayerToDeal();
        HanabiHand::CardKnowledge card_knowledge(ParentGame()->NumColors(),
                                      ParentGame()->NumRanks());
        if (parent_game_->ObservationType() == HanabiGame::kSeer){
          card_knowledge.ApplyIsColorHint(move.Color());
          card_knowledge.ApplyIsRankHint(move.Rank());
        }
        hands_[history.deal_to_player].AddCard(
            deck_.DealCard(move.Color(), move.Rank()),
            card_knowledge);
      }
      break;
    case HanabiMove::kDiscard:
      history.information_token = IncrementInformationTokens();
      history.color = hands_[cur_player_].Cards()[move.CardIndex()].Color();
      history.rank = hands_[cur_player_].Cards()[move.CardIndex()].Rank();
      hands_[cur_player_].RemoveFromHand(move.CardIndex(), &discard_pile_);
      break;
    case HanabiMove::kPlay:
      history.color = hands_[cur_player_].Cards()[move.CardIndex()].Color();
      history.rank = hands_[cur_player_].Cards()[move.CardIndex()].Rank();
      std::tie(history.scored, history.information_token) =
          AddToFireworks(hands_[cur_player_].Cards()[move.CardIndex()]);
      hands_[cur_player_].RemoveFromHand(
          move.CardIndex(), history.scored ? nullptr : &discard_pile_);
      break;
    case HanabiMove::kRevealColor:
      DecrementInformationTokens();
      history.reveal_bitmask =
          HandColorBitmask(*HandByOffset(move.TargetOffset()), move.Color());
      history.newly_revealed_bitmask =
          HandByOffset(move.TargetOffset())->RevealColor(move.Color());
      break;
    case HanabiMove::kRevealRank:
      DecrementInformationTokens();
      history.reveal_bitmask =
          HandRankBitmask(*HandByOffset(move.TargetOffset()), move.Rank());
      history.newly_revealed_bitmask =
          HandByOffset(move.TargetOffset())->RevealRank(move.Rank());
      break;
    default:
      std::abort();  // Should not be possible.
  }
  move_history_.push_back(history);
  AdvanceToNextPlayer();
}

double HanabiState::ChanceOutcomeProb(HanabiMove move) const {
  return static_cast<double>(deck_.CardCount(move.Color(), move.Rank())) /
         static_cast<double>(deck_.Size());
}

void HanabiState::ApplyRandomChance() {
  auto chance_outcomes = ChanceOutcomes();
  REQUIRE(!chance_outcomes.second.empty());
  ApplyMove(ParentGame()->PickRandomChance(chance_outcomes));
}

std::vector<HanabiMove> HanabiState::LegalMoves(int player) const {
  std::vector<HanabiMove> movelist;
  // kChancePlayer=-1 must be handled by ChanceOutcome.
  REQUIRE(player >= 0 && player < ParentGame()->NumPlayers());
  if (player != cur_player_) {
    // Turn-based game. Empty move list for other players.
    return movelist;
  }
  int max_move_uid = ParentGame()->MaxMoves();
  for (int uid = 0; uid < max_move_uid; ++uid) {
    HanabiMove move = ParentGame()->GetMove(uid);
    if (MoveIsLegal(move)) {
      movelist.push_back(move);
    }
  }
  return movelist;
}

bool HanabiState::CardPlayableOnFireworks(int color, int rank) const {
  if (color < 0 || color >= ParentGame()->NumColors()) {
    return false;
  }
  return rank == fireworks_[color];
}

std::pair<std::vector<HanabiMove>, std::vector<double>>
HanabiState::ChanceOutcomes() const {
  std::pair<std::vector<HanabiMove>, std::vector<double>> rv;
  int max_outcome_uid = ParentGame()->MaxChanceOutcomes();
  for (int uid = 0; uid < max_outcome_uid; ++uid) {
    HanabiMove move = ParentGame()->GetChanceOutcome(uid);
    if (MoveIsLegal(move)) {
      rv.first.push_back(move);
      rv.second.push_back(ChanceOutcomeProb(move));
    }
  }
  return rv;
}

// Format:  <life tokens>:<info tokens>:
//           <fireworks color 1>-....::
//            <player 1 card>-.... || <player 1 hint>-...
//            :....
//            ::<discard card 1>-...
std::string HanabiState::ToString() const {
  std::string result;
  result += "Life tokens: " + std::to_string(LifeTokens()) + "\n";
  result += "Info tokens: " + std::to_string(InformationTokens()) + "\n";
  result += "Fireworks: ";
  for (int i = 0; i < ParentGame()->NumColors(); ++i) {
    result += ColorIndexToChar(i);
    result += std::to_string(fireworks_[i]) + " ";
  }
  result += "\nHands:\n";
  for (int i = 0; i < hands_.size(); ++i) {
    if (i > 0) {
      result += "-----\n";
    }
    if (i == CurPlayer()) {
      result += "Cur player\n";
    }
    result += hands_[i].ToString();
  }
  result += "Deck size: " + std::to_string(Deck().Size()) + "\n";
  result += "Discards:";
  for (int i = 0; i < discard_pile_.size(); ++i) {
    result += " " + discard_pile_[i].ToString();
  }
  return result;
}

int HanabiState::Score() const {
  if (LifeTokens() <= 0) {
    return 0;
  }
  return std::accumulate(fireworks_.begin(), fireworks_.end(), 0);
}

HanabiState::EndOfGameType HanabiState::EndOfGameStatus() const {
  if (LifeTokens() < 1) {
    return kOutOfLifeTokens;
  }
  if (Score() >= ParentGame()->NumColors() * ParentGame()->NumRanks()) {
    return kCompletedFireworks;
  }
  if (turns_to_play_ <= 0) {
    return kOutOfCards;
  }
  return kNotFinished;
}

}  // namespace hanabi_learning_env
