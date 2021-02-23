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

#include "pyhanabi.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>

#include "hanabi_lib/canonical_encoders.h"
#include "hanabi_lib/hanabi_card.h"
#include "hanabi_lib/hanabi_game.h"
#include "hanabi_lib/hanabi_history_item.h"
#include "hanabi_lib/hanabi_move.h"
#include "hanabi_lib/hanabi_observation.h"
#include "hanabi_lib/hanabi_state.h"
#include "hanabi_lib/observation_encoder.h"
#include "hanabi_lib/util.h"

extern "C" {

/* Helpers. */

void DeleteString(char* str) { free(str); }

/* Wrapper definitions for HanabiCard. */
int CardValid(pyhanabi_card_t* card) { return card->color >= 0; }

/* Wrapper definitions for HanabiCardKnowledge. */
char* CardKnowledgeToString(pyhanabi_card_knowledge_t* knowledge) {
  REQUIRE(knowledge != nullptr);
  REQUIRE(knowledge->knowledge != nullptr);
  std::string str =
      reinterpret_cast<const hanabi_learning_env::HanabiHand::CardKnowledge*>(
          knowledge->knowledge)
          ->ToString();
  return strdup(str.c_str());
}

int ColorWasHinted(pyhanabi_card_knowledge_t* knowledge) {
  REQUIRE(knowledge != nullptr);
  REQUIRE(knowledge->knowledge != nullptr);
  return reinterpret_cast<
             const hanabi_learning_env::HanabiHand::CardKnowledge*>(
             knowledge->knowledge)
      ->ColorHinted();
}

int KnownColor(pyhanabi_card_knowledge_t* knowledge) {
  REQUIRE(knowledge != nullptr);
  REQUIRE(knowledge->knowledge != nullptr);
  return reinterpret_cast<
             const hanabi_learning_env::HanabiHand::CardKnowledge*>(
             knowledge->knowledge)
      ->Color();
}

int ColorIsPlausible(pyhanabi_card_knowledge_t* knowledge, int color) {
  REQUIRE(knowledge != nullptr);
  REQUIRE(knowledge->knowledge != nullptr);
  return reinterpret_cast<
             const hanabi_learning_env::HanabiHand::CardKnowledge*>(
             knowledge->knowledge)
      ->ColorPlausible(color);
}

int RankWasHinted(pyhanabi_card_knowledge_t* knowledge) {
  REQUIRE(knowledge != nullptr);
  REQUIRE(knowledge->knowledge != nullptr);
  return reinterpret_cast<
             const hanabi_learning_env::HanabiHand::CardKnowledge*>(
             knowledge->knowledge)
      ->RankHinted();
}

int KnownRank(pyhanabi_card_knowledge_t* knowledge) {
  REQUIRE(knowledge != nullptr);
  REQUIRE(knowledge->knowledge != nullptr);
  return reinterpret_cast<
             const hanabi_learning_env::HanabiHand::CardKnowledge*>(
             knowledge->knowledge)
      ->Rank();
}

int RankIsPlausible(pyhanabi_card_knowledge_t* knowledge, int rank) {
  REQUIRE(knowledge != nullptr);
  REQUIRE(knowledge->knowledge != nullptr);
  return reinterpret_cast<
             const hanabi_learning_env::HanabiHand::CardKnowledge*>(
             knowledge->knowledge)
      ->RankPlausible(rank);
}

/* Wrapper definitions for HanabiMove. */
void DeleteMoveList(void* movelist) {
  delete reinterpret_cast<std::vector<hanabi_learning_env::HanabiMove>*>(
      movelist);
}

int NumMoves(void* movelist) {
  return reinterpret_cast<std::vector<hanabi_learning_env::HanabiMove>*>(
             movelist)
      ->size();
}

void GetMove(void* movelist, int index, pyhanabi_move_t* move) {
  REQUIRE(move != nullptr);
  auto hanabi_movelist =
      reinterpret_cast<std::vector<hanabi_learning_env::HanabiMove>*>(movelist);
  move->move = new hanabi_learning_env::HanabiMove(hanabi_movelist->at(index));
}

void DeleteMove(pyhanabi_move_t* move) {
  REQUIRE(move != nullptr);
  REQUIRE(move->move != nullptr);
  delete reinterpret_cast<hanabi_learning_env::HanabiMove*>(move->move);
  move->move = nullptr;
}

char* MoveToString(pyhanabi_move_t* move) {
  REQUIRE(move != nullptr);
  REQUIRE(move->move != nullptr);
  std::string str =
      reinterpret_cast<const hanabi_learning_env::HanabiMove*>(move->move)
          ->ToString();
  return strdup(str.c_str());
}

int MoveType(pyhanabi_move_t* move) {
  return reinterpret_cast<const hanabi_learning_env::HanabiMove*>(move->move)
      ->MoveType();
}

int CardIndex(pyhanabi_move_t* move) {
  return reinterpret_cast<const hanabi_learning_env::HanabiMove*>(move->move)
      ->CardIndex();
}

int TargetOffset(pyhanabi_move_t* move) {
  return reinterpret_cast<const hanabi_learning_env::HanabiMove*>(move->move)
      ->TargetOffset();
}

int MoveColor(pyhanabi_move_t* move) {
  return reinterpret_cast<const hanabi_learning_env::HanabiMove*>(move->move)
      ->Color();
}

int MoveRank(pyhanabi_move_t* move) {
  return reinterpret_cast<const hanabi_learning_env::HanabiMove*>(move->move)
      ->Rank();
}

bool GetDiscardMove(int card_index, pyhanabi_move_t* move) {
  REQUIRE(move != nullptr);
  move->move = new hanabi_learning_env::HanabiMove(
      hanabi_learning_env::HanabiMove::kDiscard, card_index, -1, -1, -1);
  return move->move != nullptr;
}

bool GetPlayMove(int card_index, pyhanabi_move_t* move) {
  REQUIRE(move != nullptr);
  move->move = new hanabi_learning_env::HanabiMove(
      hanabi_learning_env::HanabiMove::kPlay, card_index, -1, -1, -1);
  return move->move != nullptr;
}

bool GetRevealColorMove(int target_offset, int color, pyhanabi_move_t* move) {
  REQUIRE(move != nullptr);
  move->move = new hanabi_learning_env::HanabiMove(
      hanabi_learning_env::HanabiMove::kRevealColor, -1, target_offset, color,
      -1);
  return move->move != nullptr;
}

bool GetRevealRankMove(int target_offset, int rank, pyhanabi_move_t* move) {
  REQUIRE(move != nullptr);
  move->move = new hanabi_learning_env::HanabiMove(
      hanabi_learning_env::HanabiMove::kRevealRank, -1, target_offset, -1,
      rank);
  return move->move != nullptr;
}

/* Wrapper definitions for HanabiHistoryItem. */
void DeleteHistoryItem(pyhanabi_history_item_t* item) {
  REQUIRE(item != nullptr);
  REQUIRE(item->item != nullptr);
  delete reinterpret_cast<hanabi_learning_env::HanabiHistoryItem*>(item->item);
  item->item = nullptr;
}

char* HistoryItemToString(pyhanabi_history_item_t* item) {
  REQUIRE(item != nullptr);
  REQUIRE(item->item != nullptr);
  std::string str =
      reinterpret_cast<const hanabi_learning_env::HanabiHistoryItem*>(
          item->item)
          ->ToString();
  return strdup(str.c_str());
}

void HistoryItemMove(pyhanabi_history_item_t* item, pyhanabi_move_t* move) {
  REQUIRE(item != nullptr);
  REQUIRE(item->item != nullptr);
  REQUIRE(move != nullptr);
  move->move = new hanabi_learning_env::HanabiMove(
      reinterpret_cast<const hanabi_learning_env::HanabiHistoryItem*>(
          item->item)
          ->move);
}

int HistoryItemPlayer(pyhanabi_history_item_t* item) {
  REQUIRE(item != nullptr);
  REQUIRE(item->item != nullptr);
  return reinterpret_cast<const hanabi_learning_env::HanabiHistoryItem*>(
             item->item)
      ->player;
}

int HistoryItemScored(pyhanabi_history_item_t* item) {
  REQUIRE(item != nullptr);
  REQUIRE(item->item != nullptr);
  return reinterpret_cast<const hanabi_learning_env::HanabiHistoryItem*>(
             item->item)
      ->scored;
}

int HistoryItemInformationToken(pyhanabi_history_item_t* item) {
  REQUIRE(item != nullptr);
  REQUIRE(item->item != nullptr);
  return reinterpret_cast<const hanabi_learning_env::HanabiHistoryItem*>(
             item->item)
      ->information_token;
}

int HistoryItemColor(pyhanabi_history_item_t* item) {
  REQUIRE(item != nullptr);
  REQUIRE(item->item != nullptr);
  return reinterpret_cast<const hanabi_learning_env::HanabiHistoryItem*>(
             item->item)
      ->color;
}

int HistoryItemRank(pyhanabi_history_item_t* item) {
  REQUIRE(item != nullptr);
  REQUIRE(item->item != nullptr);
  return reinterpret_cast<const hanabi_learning_env::HanabiHistoryItem*>(
             item->item)
      ->rank;
}

int HistoryItemRevealBitmask(pyhanabi_history_item_t* item) {
  REQUIRE(item != nullptr);
  REQUIRE(item->item != nullptr);
  return reinterpret_cast<const hanabi_learning_env::HanabiHistoryItem*>(
             item->item)
      ->reveal_bitmask;
}

int HistoryItemNewlyRevealedBitmask(pyhanabi_history_item_t* item) {
  REQUIRE(item != nullptr);
  REQUIRE(item->item != nullptr);
  return reinterpret_cast<const hanabi_learning_env::HanabiHistoryItem*>(
             item->item)
      ->newly_revealed_bitmask;
}

int HistoryItemDealToPlayer(pyhanabi_history_item_t* item) {
  REQUIRE(item != nullptr);
  REQUIRE(item->item != nullptr);
  return reinterpret_cast<const hanabi_learning_env::HanabiHistoryItem*>(
             item->item)
      ->deal_to_player;
}

/* Wrapper definitions for HanabiState. */
void NewState(pyhanabi_game_t* game, pyhanabi_state_t* state) {
  REQUIRE(state != nullptr);
  REQUIRE(game != nullptr);
  REQUIRE(game->game != nullptr);
  state->state = new hanabi_learning_env::HanabiState(
      static_cast<hanabi_learning_env::HanabiGame*>(game->game));
}

void CopyState(const pyhanabi_state_t* src, pyhanabi_state_t* dest) {
  REQUIRE(src != nullptr);
  REQUIRE(src->state != nullptr);
  REQUIRE(dest != nullptr);
  dest->state = new hanabi_learning_env::HanabiState(
      *static_cast<hanabi_learning_env::HanabiState*>(src->state));
}

void DeleteState(pyhanabi_state_t* state) {
  REQUIRE(state != nullptr);
  REQUIRE(state->state != nullptr);
  delete reinterpret_cast<hanabi_learning_env::HanabiState*>(state->state);
  state->state = nullptr;
}

const void* StateParentGame(pyhanabi_state_t* state) {
  REQUIRE(state != nullptr);
  REQUIRE(state->state != nullptr);
  return static_cast<const void*>(
      reinterpret_cast<hanabi_learning_env::HanabiState*>(state->state)
          ->ParentGame());
}

void StateApplyMove(pyhanabi_state_t* state, pyhanabi_move_t* move) {
  REQUIRE(state != nullptr);
  REQUIRE(state->state != nullptr);
  REQUIRE(move != nullptr);
  REQUIRE(move->move != nullptr);
  auto hanabi_state =
      reinterpret_cast<hanabi_learning_env::HanabiState*>(state->state);
  auto hanabi_move =
      reinterpret_cast<const hanabi_learning_env::HanabiMove*>(move->move);
  hanabi_state->ApplyMove(*hanabi_move);
}

int StateCurPlayer(pyhanabi_state_t* state) {
  REQUIRE(state != nullptr);
  REQUIRE(state->state != nullptr);
  return reinterpret_cast<hanabi_learning_env::HanabiState*>(state->state)
      ->CurPlayer();
}

void StateDealRandomCard(pyhanabi_state_t* state) {
  REQUIRE(state != nullptr);
  REQUIRE(state->state != nullptr);
  auto hanabi_state =
      reinterpret_cast<hanabi_learning_env::HanabiState*>(state->state);
  hanabi_state->ApplyRandomChance();
}

int StateDeckSize(pyhanabi_state_t* state) {
  REQUIRE(state != nullptr);
  REQUIRE(state->state != nullptr);
  return reinterpret_cast<hanabi_learning_env::HanabiState*>(state->state)
      ->Deck()
      .Size();
}

int StateFireworks(pyhanabi_state_t* state, int color) {
  REQUIRE(state != nullptr);
  REQUIRE(state->state != nullptr);
  return reinterpret_cast<hanabi_learning_env::HanabiState*>(state->state)
      ->Fireworks()
      .at(color);
}

int StateDiscardPileSize(pyhanabi_state_t* state) {
  REQUIRE(state != nullptr);
  REQUIRE(state->state != nullptr);
  return reinterpret_cast<hanabi_learning_env::HanabiState*>(state->state)
      ->DiscardPile()
      .size();
}

void StateGetDiscard(pyhanabi_state_t* state, int index,
                     pyhanabi_card_t* card) {
  REQUIRE(state != nullptr);
  REQUIRE(state->state != nullptr);
  REQUIRE(card != nullptr);
  hanabi_learning_env::HanabiCard hanabi_card =
      reinterpret_cast<hanabi_learning_env::HanabiState*>(state->state)
          ->DiscardPile()
          .at(index);
  card->color = hanabi_card.Color();
  card->rank = hanabi_card.Rank();
}

int StateGetHandSize(pyhanabi_state_t* state, int pid) {
  REQUIRE(state != nullptr);
  return reinterpret_cast<hanabi_learning_env::HanabiState*>(state->state)
      ->Hands()
      .at(pid)
      .Cards()
      .size();
}

void StateGetHandCard(pyhanabi_state_t* state, int pid, int index,
                      pyhanabi_card_t* card) {
  REQUIRE(state != nullptr);
  REQUIRE(state->state != nullptr);
  REQUIRE(card != nullptr);
  hanabi_learning_env::HanabiCard hanabi_card =
      reinterpret_cast<hanabi_learning_env::HanabiState*>(state->state)
          ->Hands()
          .at(pid)
          .Cards()
          .at(index);
  card->color = hanabi_card.Color();
  card->rank = hanabi_card.Rank();
}

int StateInformationTokens(pyhanabi_state_t* state) {
  REQUIRE(state != nullptr);
  REQUIRE(state->state != nullptr);
  return reinterpret_cast<hanabi_learning_env::HanabiState*>(state->state)
      ->InformationTokens();
}

int StateEndOfGameStatus(pyhanabi_state_t* state) {
  REQUIRE(state != nullptr);
  REQUIRE(state->state != nullptr);
  return reinterpret_cast<hanabi_learning_env::HanabiState*>(state->state)
      ->EndOfGameStatus();
}

void* StateLegalMoves(pyhanabi_state_t* state) {
  REQUIRE(state != nullptr);
  REQUIRE(state->state != nullptr);
  auto hanabi_state =
      reinterpret_cast<hanabi_learning_env::HanabiState*>(state->state);
  std::vector<hanabi_learning_env::HanabiMove>* list =
      new std::vector<hanabi_learning_env::HanabiMove>(
          hanabi_state->LegalMoves(hanabi_state->CurPlayer()));
  return static_cast<void*>(list);
}

int StateLifeTokens(pyhanabi_state_t* state) {
  REQUIRE(state != nullptr);
  REQUIRE(state->state != nullptr);
  return reinterpret_cast<hanabi_learning_env::HanabiState*>(state->state)
      ->LifeTokens();
}

int StateNumPlayers(pyhanabi_state_t* state) {
  REQUIRE(state != nullptr);
  REQUIRE(state->state != nullptr);
  return reinterpret_cast<hanabi_learning_env::HanabiState*>(state->state)
      ->ParentGame()
      ->NumPlayers();
}

int StateScore(pyhanabi_state_t* state) {
  REQUIRE(state != nullptr);
  REQUIRE(state->state != nullptr);
  return reinterpret_cast<hanabi_learning_env::HanabiState*>(state->state)
      ->Score();
}

char* StateToString(pyhanabi_state_t* state) {
  REQUIRE(state != nullptr);
  REQUIRE(state->state != nullptr);
  std::string str =
      reinterpret_cast<hanabi_learning_env::HanabiState*>(state->state)
          ->ToString();
  return strdup(str.c_str());
}

bool MoveIsLegal(const pyhanabi_state_t* state, const pyhanabi_move_t* move) {
  auto hanabi_state =
      reinterpret_cast<const hanabi_learning_env::HanabiState*>(state->state);
  auto hanabi_move =
      reinterpret_cast<const hanabi_learning_env::HanabiMove*>(move->move);
  return hanabi_state->MoveIsLegal(*hanabi_move);
}

bool CardPlayableOnFireworks(const pyhanabi_state_t* state, int color,
                             int rank) {
  return reinterpret_cast<const hanabi_learning_env::HanabiState*>(state->state)
      ->CardPlayableOnFireworks(color, rank);
}

int StateLenMoveHistory(pyhanabi_state_t* state) {
  REQUIRE(state != nullptr);
  REQUIRE(state->state != nullptr);
  return reinterpret_cast<const hanabi_learning_env::HanabiState*>(state->state)
      ->MoveHistory()
      .size();
}

void StateGetMoveHistory(pyhanabi_state_t* state, int index,
                         pyhanabi_history_item_t* item) {
  REQUIRE(state != nullptr);
  REQUIRE(state->state != nullptr);
  REQUIRE(item != nullptr);
  item->item = new hanabi_learning_env::HanabiHistoryItem(
      reinterpret_cast<const hanabi_learning_env::HanabiState*>(state->state)
          ->MoveHistory()
          .at(index));
}

/* Wrapper definitions for HanabiGame. */
void DeleteGame(pyhanabi_game_t* game) {
  REQUIRE(game != nullptr);
  REQUIRE(game->game != nullptr);
  delete reinterpret_cast<hanabi_learning_env::HanabiGame*>(game->game);
  game->game = nullptr;
}

void NewDefaultGame(pyhanabi_game_t* game) {
  std::unordered_map<std::string, std::string> params;
  REQUIRE(game != nullptr);
  game->game = static_cast<void*>(new hanabi_learning_env::HanabiGame(params));
  REQUIRE(game->game != nullptr);
}

void NewGame(pyhanabi_game_t* game, int list_length, const char** param_list) {
  std::unordered_map<std::string, std::string> params;

  for (int p = 0; p < list_length; p += 2) {
    std::string key = param_list[p];
    std::string value = param_list[p + 1];
    params[key] = value;
  }

  game->game = static_cast<hanabi_learning_env::HanabiGame*>(
      new hanabi_learning_env::HanabiGame(params));
  REQUIRE(game->game != nullptr);
}

char* GameParamString(pyhanabi_game_t* game) {
  REQUIRE(game != nullptr);
  REQUIRE(game->game != nullptr);
  std::string str;
  auto params = reinterpret_cast<hanabi_learning_env::HanabiGame*>(game->game)
                    ->Parameters();
  for (const auto& item : params) {
    str += item.first + '=' + item.second + '\n';
  }
  return strdup(str.c_str());
}

int NumPlayers(pyhanabi_game_t* game) {
  return reinterpret_cast<hanabi_learning_env::HanabiGame*>(game->game)
      ->NumPlayers();
}

int NumColors(pyhanabi_game_t* game) {
  return reinterpret_cast<hanabi_learning_env::HanabiGame*>(game->game)
      ->NumColors();
}

int NumRanks(pyhanabi_game_t* game) {
  return reinterpret_cast<hanabi_learning_env::HanabiGame*>(game->game)
      ->NumRanks();
}

int HandSize(pyhanabi_game_t* game) {
  return reinterpret_cast<hanabi_learning_env::HanabiGame*>(game->game)
      ->HandSize();
}

int MaxInformationTokens(pyhanabi_game_t* game) {
  return reinterpret_cast<hanabi_learning_env::HanabiGame*>(game->game)
      ->MaxInformationTokens();
}

int MaxLifeTokens(pyhanabi_game_t* game) {
  return reinterpret_cast<hanabi_learning_env::HanabiGame*>(game->game)
      ->MaxLifeTokens();
}

int ObservationType(pyhanabi_game_t* game) {
  return reinterpret_cast<hanabi_learning_env::HanabiGame*>(game->game)
      ->ObservationType();
}

int NumCards(pyhanabi_game_t* game, int color, int rank) {
  return reinterpret_cast<hanabi_learning_env::HanabiGame*>(game->game)
      ->NumberCardInstances(color, rank);
}

int GetMoveUid(pyhanabi_game_t* game, pyhanabi_move_t* move) {
  return reinterpret_cast<hanabi_learning_env::HanabiGame*>(game->game)
      ->GetMoveUid(*reinterpret_cast<const hanabi_learning_env::HanabiMove*>(
          move->move));
}

void GetMoveByUid(pyhanabi_game_t* game, int move_uid, pyhanabi_move_t* move) {
  REQUIRE(game != nullptr);
  REQUIRE(game->game != nullptr);
  REQUIRE(move != nullptr);
  auto hanabi_game =
      reinterpret_cast<hanabi_learning_env::HanabiGame*>(game->game);
  move->move =
      new hanabi_learning_env::HanabiMove(hanabi_game->GetMove(move_uid));
  REQUIRE(move->move != nullptr);
}

int MaxMoves(pyhanabi_game_t* game) {
  return reinterpret_cast<hanabi_learning_env::HanabiGame*>(game->game)
      ->MaxMoves();
}

/* Wrapper definitions for HanabiObservation. */
void NewObservation(pyhanabi_state_t* state, int player,
                    pyhanabi_observation_t* observation) {
  REQUIRE(state != nullptr);
  REQUIRE(state->state != nullptr);
  REQUIRE(observation != nullptr);
  observation->observation =
      static_cast<hanabi_learning_env::HanabiObservation*>(
          new hanabi_learning_env::HanabiObservation(
              *reinterpret_cast<hanabi_learning_env::HanabiState*>(
                  state->state),
              player));
  REQUIRE(observation->observation != nullptr);
}

void DeleteObservation(pyhanabi_observation_t* observation) {
  REQUIRE(observation != nullptr);
  REQUIRE(observation->observation != nullptr);
  delete reinterpret_cast<hanabi_learning_env::HanabiObservation*>(
      observation->observation);
  observation->observation = nullptr;
}

char* ObsToString(pyhanabi_observation_t* observation) {
  REQUIRE(observation != nullptr);
  REQUIRE(observation->observation != nullptr);
  std::string str = reinterpret_cast<hanabi_learning_env::HanabiObservation*>(
                        observation->observation)
                        ->ToString();
  return strdup(str.c_str());
}

int ObsCurPlayerOffset(pyhanabi_observation_t* observation) {
  REQUIRE(observation != nullptr);
  REQUIRE(observation->observation != nullptr);
  return reinterpret_cast<hanabi_learning_env::HanabiObservation*>(
             observation->observation)
      ->CurPlayerOffset();
}

int ObsNumPlayers(pyhanabi_observation_t* observation) {
  REQUIRE(observation != nullptr);
  REQUIRE(observation->observation != nullptr);
  return reinterpret_cast<hanabi_learning_env::HanabiObservation*>(
             observation->observation)
      ->ParentGame()
      ->NumPlayers();
}

int ObsGetHandSize(pyhanabi_observation_t* observation, int pid) {
  REQUIRE(observation != nullptr);
  REQUIRE(observation->observation != nullptr);
  return reinterpret_cast<hanabi_learning_env::HanabiObservation*>(
             observation->observation)
      ->Hands()
      .at(pid)
      .Cards()
      .size();
}

void ObsGetHandCard(pyhanabi_observation_t* observation, int pid, int index,
                    pyhanabi_card_t* card) {
  REQUIRE(observation != nullptr);
  REQUIRE(observation->observation != nullptr);
  REQUIRE(card != nullptr);
  hanabi_learning_env::HanabiCard hanabi_card =
      reinterpret_cast<hanabi_learning_env::HanabiObservation*>(
          observation->observation)
          ->Hands()
          .at(pid)
          .Cards()
          .at(index);
  card->color = hanabi_card.Color();
  card->rank = hanabi_card.Rank();
}

void ObsGetHandCardKnowledge(pyhanabi_observation_t* observation, int pid,
                             int index, pyhanabi_card_knowledge_t* knowledge) {
  REQUIRE(observation != nullptr);
  REQUIRE(observation->observation != nullptr);
  REQUIRE(knowledge != nullptr);
  knowledge->knowledge =
      &(reinterpret_cast<hanabi_learning_env::HanabiObservation*>(
            observation->observation)
            ->Hands()
            .at(pid)
            .Knowledge()
            .at(index));
}

int ObsDiscardPileSize(pyhanabi_observation_t* observation) {
  return reinterpret_cast<hanabi_learning_env::HanabiObservation*>(
             observation->observation)
      ->DiscardPile()
      .size();
}

void ObsGetDiscard(pyhanabi_observation_t* observation, int index,
                   pyhanabi_card_t* card) {
  REQUIRE(observation != nullptr);
  REQUIRE(observation->observation != nullptr);
  REQUIRE(card != nullptr);
  hanabi_learning_env::HanabiCard hanabi_card =
      reinterpret_cast<hanabi_learning_env::HanabiObservation*>(
          observation->observation)
          ->DiscardPile()
          .at(index);
  card->color = hanabi_card.Color();
  card->rank = hanabi_card.Rank();
}

int ObsFireworks(pyhanabi_observation_t* observation, int color) {
  REQUIRE(observation != nullptr);
  REQUIRE(observation->observation != nullptr);
  return reinterpret_cast<hanabi_learning_env::HanabiObservation*>(
             observation->observation)
      ->Fireworks()
      .at(color);
}

int ObsDeckSize(pyhanabi_observation_t* observation) {
  REQUIRE(observation != nullptr);
  REQUIRE(observation->observation != nullptr);
  return reinterpret_cast<hanabi_learning_env::HanabiObservation*>(
             observation->observation)
      ->DeckSize();
}

int ObsNumLastMoves(pyhanabi_observation_t* observation) {
  REQUIRE(observation != nullptr);
  REQUIRE(observation->observation != nullptr);
  return reinterpret_cast<hanabi_learning_env::HanabiObservation*>(
             observation->observation)
      ->LastMoves()
      .size();
}

void ObsGetLastMove(pyhanabi_observation_t* observation, int index,
                    pyhanabi_history_item_t* item) {
  REQUIRE(observation != nullptr);
  REQUIRE(observation->observation != nullptr);
  REQUIRE(item != nullptr);
  item->item = new hanabi_learning_env::HanabiHistoryItem(
      reinterpret_cast<hanabi_learning_env::HanabiObservation*>(
          observation->observation)
          ->LastMoves()
          .at(index));
}

int ObsInformationTokens(pyhanabi_observation_t* observation) {
  REQUIRE(observation != nullptr);
  REQUIRE(observation->observation != nullptr);
  return reinterpret_cast<hanabi_learning_env::HanabiObservation*>(
             observation->observation)
      ->InformationTokens();
}

int ObsLifeTokens(pyhanabi_observation_t* observation) {
  REQUIRE(observation != nullptr);
  REQUIRE(observation->observation != nullptr);
  return reinterpret_cast<hanabi_learning_env::HanabiObservation*>(
             observation->observation)
      ->LifeTokens();
}

int ObsNumLegalMoves(pyhanabi_observation_t* observation) {
  REQUIRE(observation != nullptr);
  REQUIRE(observation->observation != nullptr);
  return reinterpret_cast<hanabi_learning_env::HanabiObservation*>(
             observation->observation)
      ->LegalMoves()
      .size();
}

void ObsGetLegalMove(pyhanabi_observation_t* observation, int index,
                     pyhanabi_move_t* move) {
  REQUIRE(observation != nullptr);
  REQUIRE(observation->observation != nullptr);
  REQUIRE(move != nullptr);
  move->move = new hanabi_learning_env::HanabiMove(
      (reinterpret_cast<hanabi_learning_env::HanabiObservation*>(
           observation->observation)
           ->LegalMoves()
           .at(index)));
}

bool ObsCardPlayableOnFireworks(const pyhanabi_observation_t* observation,
                                int color, int rank) {
  return reinterpret_cast<const hanabi_learning_env::HanabiObservation*>(
             observation->observation)
      ->CardPlayableOnFireworks(color, rank);
}

void NewObservationEncoder(pyhanabi_observation_encoder_t* encoder,
                           pyhanabi_game_t* game, int type) {
  REQUIRE(encoder != nullptr);
  REQUIRE(game != nullptr);
  REQUIRE(game->game != nullptr);
  auto otype = static_cast<hanabi_learning_env::ObservationEncoder::Type>(type);
  auto hanabi_game =
      reinterpret_cast<hanabi_learning_env::HanabiGame*>(game->game);
  switch (otype) {
    case hanabi_learning_env::ObservationEncoder::Type::kCanonical:
      encoder->encoder = static_cast<hanabi_learning_env::ObservationEncoder*>(
          new hanabi_learning_env::CanonicalObservationEncoder(hanabi_game));
      break;
    default:
      std::cerr << "Encoder type not recognized." << std::endl;
      encoder->encoder = nullptr;
      std::abort();
  }
}

void DeleteObservationEncoder(pyhanabi_observation_encoder_t* encoder) {
  REQUIRE(encoder != nullptr);
  REQUIRE(encoder->encoder != nullptr);
  delete reinterpret_cast<hanabi_learning_env::ObservationEncoder*>(
      encoder->encoder);
  encoder->encoder = nullptr;
}

char* ObservationShape(pyhanabi_observation_encoder_t* encoder) {
  REQUIRE(encoder != nullptr);
  REQUIRE(encoder->encoder != nullptr);
  auto obs_enc = reinterpret_cast<hanabi_learning_env::ObservationEncoder*>(
      encoder->encoder);
  std::vector<int> shape = obs_enc->Shape();
  std::string shape_str = "";
  for (int i = 0; i < shape.size(); i++) {
    shape_str += std::to_string(shape[i]);
    if (i != shape.size() - 1) {
      shape_str += ",";
    }
  }
  return strdup(shape_str.c_str());
}

char* OwnHandShape(pyhanabi_observation_encoder_t* encoder) {
  REQUIRE(encoder != nullptr);
  REQUIRE(encoder->encoder != nullptr);
  auto obs_enc = reinterpret_cast<hanabi_learning_env::ObservationEncoder*>(
      encoder->encoder);
  std::vector<int> shape = obs_enc->OwnHandShape();
  std::string shape_str = "";
  for (int i = 0; i < shape.size(); i++) {
    shape_str += std::to_string(shape[i]);
    if (i != shape.size() - 1) {
      shape_str += ",";
    }
  }
  return strdup(shape_str.c_str());
}

char* EncodeObservation(pyhanabi_observation_encoder_t* encoder,
                        pyhanabi_observation_t* observation) {
  REQUIRE(encoder != nullptr);
  REQUIRE(encoder->encoder != nullptr);
  REQUIRE(observation != nullptr);
  REQUIRE(observation->observation != nullptr);
  auto obs_enc = reinterpret_cast<hanabi_learning_env::ObservationEncoder*>(
      encoder->encoder);
  auto obs = reinterpret_cast<hanabi_learning_env::HanabiObservation*>(
      observation->observation);
  std::vector<int> encoding = obs_enc->Encode(*obs);
  std::string obs_str = "";
  for (int i = 0; i < encoding.size(); i++) {
    obs_str += (encoding[i] ? "1" : "0");
    if (i != encoding.size() - 1) {
      obs_str += ",";
    }
  }
  return strdup(obs_str.c_str());
}

char* EncodeOwnHandObservation(pyhanabi_observation_encoder_t* encoder,
                        pyhanabi_observation_t* observation) {
  REQUIRE(encoder != nullptr);
  REQUIRE(encoder->encoder != nullptr);
  REQUIRE(observation != nullptr);
  REQUIRE(observation->observation != nullptr);
  auto obs_enc = reinterpret_cast<hanabi_learning_env::ObservationEncoder*>(
      encoder->encoder);
  auto obs = reinterpret_cast<hanabi_learning_env::HanabiObservation*>(
      observation->observation);
  std::vector<int> encoding = obs_enc->EncodeOwnHand(*obs);
  std::string obs_str = "";
  for (int i = 0; i < encoding.size(); i++) {
    obs_str += (encoding[i] ? "1" : "0");
    if (i != encoding.size() - 1) {
      obs_str += ",";
    }
  }
  return strdup(obs_str.c_str());
}

} /* extern "C" */
