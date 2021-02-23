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

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "canonical_encoders.h"

namespace hanabi_learning_env {

namespace {

// Computes the product of dimensions in shape, i.e. how many individual
// pieces of data the encoded observation requires.
int FlatLength(const std::vector<int>& shape) {
  return std::accumulate(std::begin(shape), std::end(shape), 1,
                         std::multiplies<int>());
}

const HanabiHistoryItem* GetLastNonDealMove(
    const std::vector<HanabiHistoryItem>& past_moves) {
  auto it = std::find_if(
      past_moves.begin(), past_moves.end(), [](const HanabiHistoryItem& item) {
        return item.move.MoveType() != HanabiMove::Type::kDeal;
      });
  return it == past_moves.end() ? nullptr : &(*it);
}

int BitsPerCard(const HanabiGame& game) {
  return game.NumColors() * game.NumRanks();
}

// The card's one-hot index using a color-major ordering.
int CardIndex(int color, int rank, int num_ranks) {
  return color * num_ranks + rank;
}

int HandsSectionLength(const HanabiGame& game) {
  return (game.NumPlayers() - 1) * game.HandSize() * BitsPerCard(game) +
         game.NumPlayers();
}

int OwnHandLength(const HanabiGame& game) {
  return game.HandSize() * BitsPerCard(game);
}

// Enocdes cards in all other player's hands (excluding our unknown hand),
// and whether the hand is missing a card for all players (when deck is empty.)
// Each card in a hand is encoded with a one-hot representation using
// <num_colors> * <num_ranks> bits (25 bits in a standard game) per card.
// Returns the number of entries written to the encoding.
int EncodeHands(const HanabiGame& game, const HanabiObservation& obs,
                int start_offset, std::vector<int>* encoding) {
  int bits_per_card = BitsPerCard(game);
  int num_ranks = game.NumRanks();
  int num_players = game.NumPlayers();
  int hand_size = game.HandSize();

  int offset = start_offset;
  const std::vector<HanabiHand>& hands = obs.Hands();
  assert(hands.size() == num_players);
  for (int player = 1; player < num_players; ++player) {
    const std::vector<HanabiCard>& cards = hands[player].Cards();
    int num_cards = 0;

    for (const HanabiCard& card : cards) {
      // Only a player's own cards can be invalid/unobserved.
      assert(card.IsValid());
      assert(card.Color() < game.NumColors());
      assert(card.Rank() < num_ranks);
      (*encoding)[offset + CardIndex(card.Color(), card.Rank(), num_ranks)] = 1;

      ++num_cards;
      offset += bits_per_card;
    }

    // A player's hand can have fewer cards than the initial hand size.
    // Leave the bits for the absent cards empty (adjust the offset to skip
    // bits for the missing cards).
    if (num_cards < hand_size) {
      offset += (hand_size - num_cards) * bits_per_card;
    }
  }

  // For each player, set a bit if their hand is missing a card.
  for (int player = 0; player < num_players; ++player) {
    if (hands[player].Cards().size() < game.HandSize()) {
      (*encoding)[offset + player] = 1;
    }
  }
  offset += num_players;

  assert(offset - start_offset == HandsSectionLength(game));
  return offset - start_offset;
}

int BoardSectionLength(const HanabiGame& game) {
  return game.MaxDeckSize() - game.NumPlayers() * game.HandSize() +  // deck
         game.NumColors() * game.NumRanks() +  // fireworks
         game.MaxInformationTokens() +         // info tokens
         game.MaxLifeTokens();                 // life tokens
}

// Encode the board, including:
//   - remaining deck size
//     (max_deck_size - num_players * hand_size bits; thermometer)
//   - state of the fireworks (<num_ranks> bits per color; one-hot)
//   - information tokens remaining (max_information_tokens bits; thermometer)
//   - life tokens remaining (max_life_tokens bits; thermometer)
// We note several features use a thermometer representation instead of one-hot.
// For example, life tokens could be: 000 (0), 100 (1), 110 (2), 111 (3).
// Returns the number of entries written to the encoding.
int EncodeBoard(const HanabiGame& game, const HanabiObservation& obs,
                int start_offset, std::vector<int>* encoding) {
  int num_colors = game.NumColors();
  int num_ranks = game.NumRanks();
  int num_players = game.NumPlayers();
  int hand_size = game.HandSize();
  int max_deck_size = game.MaxDeckSize();

  int offset = start_offset;
  // Encode the deck size
  for (int i = 0; i < obs.DeckSize(); ++i) {
    (*encoding)[offset + i] = 1;
  }
  offset += (max_deck_size - hand_size * num_players);  // 40 in normal 2P game

  // fireworks
  const std::vector<int>& fireworks = obs.Fireworks();
  for (int c = 0; c < num_colors; ++c) {
    // fireworks[color] is the number of successfully played <color> cards.
    // If some were played, one-hot encode the highest (0-indexed) rank played
    if (fireworks[c] > 0) {
      (*encoding)[offset + fireworks[c] - 1] = 1;
    }
    offset += num_ranks;
  }

  // info tokens
  assert(obs.InformationTokens() >= 0);
  assert(obs.InformationTokens() <= game.MaxInformationTokens());
  for (int i = 0; i < obs.InformationTokens(); ++i) {
    (*encoding)[offset + i] = 1;
  }
  offset += game.MaxInformationTokens();

  // life tokens
  assert(obs.LifeTokens() >= 0);
  assert(obs.LifeTokens() <= game.MaxLifeTokens());
  for (int i = 0; i < obs.LifeTokens(); ++i) {
    (*encoding)[offset + i] = 1;
  }
  offset += game.MaxLifeTokens();

  assert(offset - start_offset == BoardSectionLength(game));
  return offset - start_offset;
}

int DiscardSectionLength(const HanabiGame& game) { return game.MaxDeckSize(); }

// Encode the discard pile. (max_deck_size bits)
// Encoding is in color-major ordering, as in kColorStr ("RYGWB"), with each
// color and rank using a thermometer to represent the number of cards
// discarded. For example, in a standard game, there are 3 cards of lowest rank
// (1), 1 card of highest rank (5), 2 of all else. So each color would be
// ordered like so:
//
//   LLL      H
//   1100011101
//
// This means for this color:
//   - 2 cards of the lowest rank have been discarded
//   - none of the second lowest rank have been discarded
//   - both of the third lowest rank have been discarded
//   - one of the second highest rank have been discarded
//   - the highest rank card has been discarded
// Returns the number of entries written to the encoding.
int EncodeDiscards(const HanabiGame& game, const HanabiObservation& obs,
                   int start_offset, std::vector<int>* encoding) {
  int num_colors = game.NumColors();
  int num_ranks = game.NumRanks();

  int offset = start_offset;
  std::vector<int> discard_counts(num_colors * num_ranks, 0);
  for (const HanabiCard& card : obs.DiscardPile()) {
    ++discard_counts[card.Color() * num_ranks + card.Rank()];
  }

  for (int c = 0; c < num_colors; ++c) {
    for (int r = 0; r < num_ranks; ++r) {
      int num_discarded = discard_counts[c * num_ranks + r];
      for (int i = 0; i < num_discarded; ++i) {
        (*encoding)[offset + i] = 1;
      }
      offset += game.NumberCardInstances(c, r);
    }
  }

  assert(offset - start_offset == DiscardSectionLength(game));
  return offset - start_offset;
}

int LastActionSectionLength(const HanabiGame& game) {
  return game.NumPlayers() +  // player id
         4 +                  // move types (play, dis, rev col, rev rank)
         game.NumPlayers() +  // target player id (if hint action)
         game.NumColors() +   // color (if hint action)
         game.NumRanks() +    // rank (if hint action)
         game.HandSize() +    // outcome (if hint action)
         game.HandSize() +    // position (if play action)
         BitsPerCard(game) +  // card (if play or discard action)
         2;                   // play (successful, added information token)
}

// Encode the last player action (not chance's deal of cards). This encodes:
//  - Acting player index, relative to ourself (<num_players> bits; one-hot)
//  - The MoveType (4 bits; one-hot)
//  - Target player index, relative to acting player, if a reveal move
//    (<num_players> bits; one-hot)
//  - Color revealed, if a reveal color move (<num_colors> bits; one-hot)
//  - Rank revealed, if a reveal rank move (<num_ranks> bits; one-hot)
//  - Reveal outcome (<hand_size> bits; each bit is 1 if the card was hinted at)
//  - Position played/discarded (<hand_size> bits; one-hot)
//  - Card played/discarded (<num_colors> * <num_ranks> bits; one-hot)
// Returns the number of entries written to the encoding.
int EncodeLastAction(const HanabiGame& game, const HanabiObservation& obs,
                     int start_offset, std::vector<int>* encoding) {
  int num_colors = game.NumColors();
  int num_ranks = game.NumRanks();
  int num_players = game.NumPlayers();
  int hand_size = game.HandSize();

  int offset = start_offset;
  const HanabiHistoryItem* last_move = GetLastNonDealMove(obs.LastMoves());
  if (last_move == nullptr) {
    offset += LastActionSectionLength(game);
  } else {
    HanabiMove::Type last_move_type = last_move->move.MoveType();

    // player_id
    // Note: no assertion here. At a terminal state, the last player could have
    // been me (player id 0).
    (*encoding)[offset + last_move->player] = 1;
    offset += num_players;

    // move type
    switch (last_move_type) {
      case HanabiMove::Type::kPlay:
        (*encoding)[offset] = 1;
        break;
      case HanabiMove::Type::kDiscard:
        (*encoding)[offset + 1] = 1;
        break;
      case HanabiMove::Type::kRevealColor:
        (*encoding)[offset + 2] = 1;
        break;
      case HanabiMove::Type::kRevealRank:
        (*encoding)[offset + 3] = 1;
        break;
      default:
        std::abort();
    }
    offset += 4;

    // target player (if hint action)
    if (last_move_type == HanabiMove::Type::kRevealColor ||
        last_move_type == HanabiMove::Type::kRevealRank) {
      int8_t observer_relative_target =
          (last_move->player + last_move->move.TargetOffset()) % num_players;
      (*encoding)[offset + observer_relative_target] = 1;
    }
    offset += num_players;

    // color (if hint action)
    if (last_move_type == HanabiMove::Type::kRevealColor) {
      (*encoding)[offset + last_move->move.Color()] = 1;
    }
    offset += num_colors;

    // rank (if hint action)
    if (last_move_type == HanabiMove::Type::kRevealRank) {
      (*encoding)[offset + last_move->move.Rank()] = 1;
    }
    offset += num_ranks;

    // outcome (if hinted action)
    if (last_move_type == HanabiMove::Type::kRevealColor ||
        last_move_type == HanabiMove::Type::kRevealRank) {
      for (int i = 0, mask = 1; i < hand_size; ++i, mask <<= 1) {
        if ((last_move->reveal_bitmask & mask) > 0) {
          (*encoding)[offset + i] = 1;
        }
      }
    }
    offset += hand_size;

    // position (if play or discard action)
    if (last_move_type == HanabiMove::Type::kPlay ||
        last_move_type == HanabiMove::Type::kDiscard) {
      (*encoding)[offset + last_move->move.CardIndex()] = 1;
    }
    offset += hand_size;

    // card (if play or discard action)
    if (last_move_type == HanabiMove::Type::kPlay ||
        last_move_type == HanabiMove::Type::kDiscard) {
      assert(last_move->color >= 0);
      assert(last_move->rank >= 0);
      (*encoding)[offset +
                  CardIndex(last_move->color, last_move->rank, num_ranks)] = 1;
    }
    offset += BitsPerCard(game);

    // was successful and/or added information token (if play action)
    if (last_move_type == HanabiMove::Type::kPlay) {
      if (last_move->scored) {
        (*encoding)[offset] = 1;
      }
      if (last_move->information_token) {
        (*encoding)[offset + 1] = 1;
      }
    }
    offset += 2;
  }

  assert(offset - start_offset == LastActionSectionLength(game));
  return offset - start_offset;
}

int CardKnowledgeSectionLength(const HanabiGame& game) {
  return game.NumPlayers() * game.HandSize() *
         (BitsPerCard(game) + game.NumColors() + game.NumRanks());
}

int V0BeliefSectionLength(const HanabiGame& game) {
  int hand_size = game.HandSize();
  int num_players = game.NumPlayers();
  return num_players * hand_size *
         (BitsPerCard(game) + game.NumColors() + game.NumRanks());
}

// Encode the common card knowledge.
// For each card/position in each player's hand, including the observing player,
// encode the possible cards that could be in that position and whether the
// color and rank were directly revealed by a Reveal action. Possible card
// values are in color-major order, using <num_colors> * <num_ranks> bits per
// card. For example, if you knew nothing about a card, and a player revealed
// that is was green, the knowledge would be encoded as follows.
// R    Y    G    W    B
// 0000000000111110000000000   Only green cards are possible.
// 0    0    1    0    0       Card was revealed to be green.
// 00000                       Card rank was not revealed.
//
// Similarly, if the player revealed that one of your other cards was green, you
// would know that this card could not be green, resulting in:
// R    Y    G    W    B
// 1111111111000001111111111   Any card that is not green is possible.
// 0    0    0    0    0       Card color was not revealed.
// 00000                       Card rank was not revealed.
// Uses <num_players> * <hand_size> *
// (<num_colors> * <num_ranks> + <num_colors> + <num_ranks>) bits.
// Returns the number of entries written to the encoding.
int EncodeCardKnowledge(const HanabiGame& game, const HanabiObservation& obs,
                        int start_offset, std::vector<int>* encoding) {
  int bits_per_card = BitsPerCard(game);
  int num_colors = game.NumColors();
  int num_ranks = game.NumRanks();
  int num_players = game.NumPlayers();
  int hand_size = game.HandSize();

  int offset = start_offset;
  const std::vector<HanabiHand>& hands = obs.Hands();
  assert(hands.size() == num_players);
  for (int player = 0; player < num_players; ++player) {
    const std::vector<HanabiHand::CardKnowledge>& knowledge =
        hands[player].Knowledge();
    int num_cards = 0;

    for (const HanabiHand::CardKnowledge& card_knowledge : knowledge) {
      // Add bits for plausible card.
      for (int color = 0; color < num_colors; ++color) {
        if (card_knowledge.ColorPlausible(color)) {
          for (int rank = 0; rank < num_ranks; ++rank) {
            if (card_knowledge.RankPlausible(rank)) {
              (*encoding)[offset + CardIndex(color, rank, num_ranks)] = 1;
            }
          }
        }
      }
      offset += bits_per_card;

      // Add bits for explicitly revealed colors and ranks.
      if (card_knowledge.ColorHinted()) {
        (*encoding)[offset + card_knowledge.Color()] = 1;
      }
      offset += num_colors;
      if (card_knowledge.RankHinted()) {
        (*encoding)[offset + card_knowledge.Rank()] = 1;
      }
      offset += num_ranks;

      ++num_cards;
    }

    // A player's hand can have fewer cards than the initial hand size.
    // Leave the bits for the absent cards empty (adjust the offset to skip
    // bits for the missing cards).
    if (num_cards < hand_size) {
      offset +=
          (hand_size - num_cards) * (bits_per_card + num_colors + num_ranks);
    }
  }

  assert(offset - start_offset == CardKnowledgeSectionLength(game));
  return offset - start_offset;
}

std::vector<int> ComputeCardCount(const HanabiGame& game,
                                  const HanabiObservation& obs) {
  int num_colors = game.NumColors();
  int num_ranks = game.NumRanks();

  std::vector<int> card_count(num_colors * num_ranks, 0);
  int total_count = 0;
  // full deck card count
  for (int color = 0; color < game.NumColors(); ++color) {
    for (int rank = 0; rank < game.NumRanks(); ++rank) {
      auto count = game.NumberCardInstances(color, rank);
      card_count[CardIndex(color, rank, num_ranks)] = count;
      total_count += count;
    }
  }
  // remove discard
  for (const HanabiCard& card : obs.DiscardPile()) {
    --card_count[CardIndex(card.Color(), card.Rank(), num_ranks)];
    --total_count;
  }
  // remove fireworks on board
  const std::vector<int>& fireworks = obs.Fireworks();
  for (int c = 0; c < num_colors; ++c) {
    // fireworks[color] is the number of successfully played <color> cards.
    // If some were played, one-hot encode the highest (0-indexed) rank played
    if (fireworks[c] > 0) {
      for (int rank = 0; rank < fireworks[c]; ++rank) {
        --card_count[CardIndex(c, rank, num_ranks)];
        --total_count;
      }
    }
  }

  {
    // sanity check
    const std::vector<HanabiHand>& hands = obs.Hands();
    int total_hand_size = 0;
    for (const auto& hand : hands) {
      total_hand_size += hand.Cards().size();
    }
    if(total_count != obs.DeckSize() + total_hand_size) {
      std::cout << "size mismatch: " << total_count
                << " vs " << obs.DeckSize() + total_hand_size << std::endl;
      assert(false);
    }
  }
  return card_count;
}

int EncodeV0Belief_(const HanabiGame& game,
                    const HanabiObservation& obs,
                    int start_offset,
                    std::vector<int>* encoding) 
                    {
  // int bits_per_card = BitsPerCard(game);
  int num_colors = game.NumColors();
  int num_ranks = game.NumRanks();
  int num_players = game.NumPlayers();
  int hand_size = game.HandSize();

  std::vector<int> card_count = ComputeCardCount(game, obs);

  // card knowledge
  const int len = EncodeCardKnowledge(game, obs, start_offset, encoding);
  const int player_offset = len / num_players;
  const int per_card_offset = len / hand_size / num_players;
  assert(per_card_offset == num_colors * num_ranks + num_colors + num_ranks);

  const std::vector<HanabiHand>& hands = obs.Hands();
  for (int player_id = 0; player_id < num_players; ++player_id) {
    int num_cards = hands[player_id].Cards().size();
    for (int card_idx = 0; card_idx < num_cards; ++card_idx) {
      float total = 0;
      for (int i = 0; i < num_colors * num_ranks; ++i) {
        int offset = (start_offset
                      + player_offset * player_id
                      + card_idx * per_card_offset
                      + i);
        // std::cout << offset << ", " << len << std::endl;
        assert(offset - start_offset < len);
        (*encoding)[offset] *= card_count[i];
        total += (*encoding)[offset];
      }
      if (total <= 0) {
        // const std::vector<HanabiHand>& hands = obs.Hands();
        std::cout << hands[0].Cards().size() << std::endl;
        std::cout << hands[1].Cards().size() << std::endl;
        std::cout << "total = 0 " << std::endl;
        assert(false);
      }
      for (int i = 0; i < num_colors * num_ranks; ++i) {
        int offset = (start_offset
                      + player_offset * player_id
                      + card_idx * per_card_offset
                      + i);
        (*encoding)[offset] /= total;
      }
    }
  }

  assert(len == V0BeliefSectionLength(game));
  return len;
}

}  // namespace

std::vector<int> CanonicalObservationEncoder::Shape() const {
    int l = HandsSectionLength(*parent_game_) +
          BoardSectionLength(*parent_game_) +
          DiscardSectionLength(*parent_game_) +
          LastActionSectionLength(*parent_game_) +
          (parent_game_->ObservationType() == HanabiGame::kMinimal
               ? 0
               : V0BeliefSectionLength(*parent_game_));
  return {l};
}

std::vector<int> CanonicalObservationEncoder::OwnHandShape() const {
  return {OwnHandLength(*parent_game_)};
}

std::vector<int> CanonicalObservationEncoder::Encode(const HanabiObservation& obs) const {
  // Make an empty bit string of the proper size.
  std::vector<int> encoding(FlatLength(Shape()), 0);

  // This offset is an index to the start of each section of the bit vector.
  // It is incremented at the end of each section.
  int offset = 0;
  offset += EncodeHands(*parent_game_, obs, offset, &encoding);
  offset += EncodeBoard(*parent_game_, obs, offset, &encoding);
  offset += EncodeDiscards(*parent_game_, obs, offset, &encoding);
  offset += EncodeLastAction(*parent_game_, obs, offset, &encoding);

  if (parent_game_->ObservationType() != HanabiGame::kMinimal) {
    offset += EncodeV0Belief_(*parent_game_, obs, offset, &encoding);
  }

  assert(offset == encoding.size());
  return encoding;
}

std::vector<int> CanonicalObservationEncoder::EncodeOwnHand(
    const HanabiObservation& obs) const {
  int bits_per_card =  BitsPerCard(*parent_game_);
  int len = parent_game_->HandSize() * bits_per_card;
  std::vector<int> encoding(len, 0);

  const std::vector<HanabiCard>& cards = obs.OwnHands()[0].Cards();
  const int num_ranks = parent_game_->NumRanks();

  int offset = 0;
  for (const HanabiCard& card : cards) {
    // Only a player's own cards can be invalid/unobserved.
    assert(card.IsValid());
    int idx = CardIndex(card.Color(), card.Rank(), num_ranks);
    encoding[offset + idx] = 1;
    offset += bits_per_card;
  }

  assert(offset == cards.size() * bits_per_card);
  return encoding;
}

}  // namespace hanabi_learning_env
