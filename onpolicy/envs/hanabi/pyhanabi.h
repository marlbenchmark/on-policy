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

#ifndef __PYHANABI_H__
#define __PYHANABI_H__

/**
 * This is a pure C API to the C++ code.
 * All the declarations are loaded in pyhanabi.py.
 * The set of functions below is referred to as the 'cdef' throughout the code.
 */

extern "C" {

typedef struct PyHanabiCard {
  int color;
  int rank;
} pyhanabi_card_t;

typedef struct PyHanabiCardKnowledge {
  /* Points to a hanabi_learning_env::HanabiHand::CardKnowledge. */
  const void* knowledge;
} pyhanabi_card_knowledge_t;

typedef struct PyHanabiMove {
  /* Points to a hanabi_learning_env::HanabiMove. */
  void* move;
} pyhanabi_move_t;

typedef struct PyHanabiHistoryItem {
  /* Points to a hanabi_learning_env::HanabiHistoryItem. */
  void* item;
} pyhanabi_history_item_t;

typedef struct PyHanabiState {
  /* Points to a hanabi_learning_env::HanabiState. */
  void* state;
} pyhanabi_state_t;

typedef struct PyHanabiGame {
  /* Points to a hanabi_learning_env::HanabiGame. */
  void* game;
} pyhanabi_game_t;

typedef struct PyHanabiObservation {
  /* Points to a hanabi_learning_env::HanabiObservation. */
  void* observation;
} pyhanabi_observation_t;

typedef struct PyHanabiObservationEncoder {
  /* Points to a hanabi_learning_env::ObservationEncoder. */
  void* encoder;
} pyhanabi_observation_encoder_t;

/* Utility Functions. */
void DeleteString(char* str);

/* Card functions. */
int CardValid(pyhanabi_card_t* card);

/* CardKnowledge functions */
char* CardKnowledgeToString(pyhanabi_card_knowledge_t* knowledge);
int ColorWasHinted(pyhanabi_card_knowledge_t* knowledge);
int KnownColor(pyhanabi_card_knowledge_t* knowledge);
int ColorIsPlausible(pyhanabi_card_knowledge_t* knowledge, int color);
int RankWasHinted(pyhanabi_card_knowledge_t* knowledge);
int KnownRank(pyhanabi_card_knowledge_t* knowledge);
int RankIsPlausible(pyhanabi_card_knowledge_t* knowledge, int rank);

/* Move functions. */
void DeleteMoveList(void* movelist);
int NumMoves(void* movelist);
void GetMove(void* movelist, int index, pyhanabi_move_t* move);
void DeleteMove(pyhanabi_move_t* move);
char* MoveToString(pyhanabi_move_t* move);
int MoveType(pyhanabi_move_t* move);
int CardIndex(pyhanabi_move_t* move);
int TargetOffset(pyhanabi_move_t* move);
int MoveColor(pyhanabi_move_t* move);
int MoveRank(pyhanabi_move_t* move);
bool GetDiscardMove(int card_index, pyhanabi_move_t* move);
bool GetPlayMove(int card_index, pyhanabi_move_t* move);
bool GetRevealColorMove(int target_offset, int color, pyhanabi_move_t* move);
bool GetRevealRankMove(int target_offset, int rank, pyhanabi_move_t* move);

/* HistoryItem functions. */
void DeleteHistoryItem(pyhanabi_history_item_t* item);
char* HistoryItemToString(pyhanabi_history_item_t* item);
void HistoryItemMove(pyhanabi_history_item_t* item, pyhanabi_move_t* move);
int HistoryItemPlayer(pyhanabi_history_item_t* item);
int HistoryItemScored(pyhanabi_history_item_t* item);
int HistoryItemInformationToken(pyhanabi_history_item_t* item);
int HistoryItemColor(pyhanabi_history_item_t* item);
int HistoryItemRank(pyhanabi_history_item_t* item);
int HistoryItemRevealBitmask(pyhanabi_history_item_t* item);
int HistoryItemNewlyRevealedBitmask(pyhanabi_history_item_t* item);
int HistoryItemDealToPlayer(pyhanabi_history_item_t* item);

/* State functions. */
void NewState(pyhanabi_game_t* game, pyhanabi_state_t* state);
void CopyState(const pyhanabi_state_t* src, pyhanabi_state_t* dest);
void DeleteState(pyhanabi_state_t* state);
const void* StateParentGame(pyhanabi_state_t* state);
void StateApplyMove(pyhanabi_state_t* state, pyhanabi_move_t* move);
int StateCurPlayer(pyhanabi_state_t* state);
void StateDealRandomCard(pyhanabi_state_t* state);
int StateDeckSize(pyhanabi_state_t* state);
int StateFireworks(pyhanabi_state_t* state, int color);
int StateDiscardPileSize(pyhanabi_state_t* state);
void StateGetDiscard(pyhanabi_state_t* state, int index, pyhanabi_card_t* card);
int StateGetHandSize(pyhanabi_state_t* state, int pid);
void StateGetHandCard(pyhanabi_state_t* state, int pid, int index,
                      pyhanabi_card_t* card);
int StateEndOfGameStatus(pyhanabi_state_t* state);
int StateInformationTokens(pyhanabi_state_t* state);
void* StateLegalMoves(pyhanabi_state_t* state);
int StateLifeTokens(pyhanabi_state_t* state);
int StateNumPlayers(pyhanabi_state_t* state);
int StateScore(pyhanabi_state_t* state);
char* StateToString(pyhanabi_state_t* state);
bool MoveIsLegal(const pyhanabi_state_t* state, const pyhanabi_move_t* move);
bool CardPlayableOnFireworks(const pyhanabi_state_t* state, int color,
                             int rank);
int StateLenMoveHistory(pyhanabi_state_t* state);
void StateGetMoveHistory(pyhanabi_state_t* state, int index,
                         pyhanabi_history_item_t* item);

/* Game functions. */
void DeleteGame(pyhanabi_game_t* game);
void NewDefaultGame(pyhanabi_game_t* game);
void NewGame(pyhanabi_game_t* game, int list_length, const char** param_list);
char* GameParamString(pyhanabi_game_t* game);
int NumPlayers(pyhanabi_game_t* game);
int NumColors(pyhanabi_game_t* game);
int NumRanks(pyhanabi_game_t* game);
int HandSize(pyhanabi_game_t* game);
int MaxInformationTokens(pyhanabi_game_t* game);
int MaxLifeTokens(pyhanabi_game_t* game);
int ObservationType(pyhanabi_game_t* game);
int NumCards(pyhanabi_game_t* game, int color, int rank);
int GetMoveUid(pyhanabi_game_t* game, pyhanabi_move_t* move);
void GetMoveByUid(pyhanabi_game_t* game, int move_uid, pyhanabi_move_t* move);
int MaxMoves(pyhanabi_game_t* game);

/* Observation functions. */
void NewObservation(pyhanabi_state_t* state, int player,
                    pyhanabi_observation_t* observation);
void DeleteObservation(pyhanabi_observation_t* observation);
char* ObsToString(pyhanabi_observation_t* observation);
int ObsCurPlayerOffset(pyhanabi_observation_t* observation);
int ObsNumPlayers(pyhanabi_observation_t* observation);
int ObsGetHandSize(pyhanabi_observation_t* observation, int pid);
void ObsGetHandCard(pyhanabi_observation_t* observation, int pid, int index,
                    pyhanabi_card_t* card);
void ObsGetHandCardKnowledge(pyhanabi_observation_t* observation, int pid,
                             int index, pyhanabi_card_knowledge_t* knowledge);
int ObsDiscardPileSize(pyhanabi_observation_t* observation);
void ObsGetDiscard(pyhanabi_observation_t* observation, int index,
                   pyhanabi_card_t* card);
int ObsFireworks(pyhanabi_observation_t* observation, int color);
int ObsDeckSize(pyhanabi_observation_t* observation);
int ObsNumLastMoves(pyhanabi_observation_t* observation);
void ObsGetLastMove(pyhanabi_observation_t* observation, int index,
                    pyhanabi_history_item_t* item);
int ObsInformationTokens(pyhanabi_observation_t* observation);
int ObsLifeTokens(pyhanabi_observation_t* observation);
int ObsNumLegalMoves(pyhanabi_observation_t* observation);
void ObsGetLegalMove(pyhanabi_observation_t* observation, int index,
                     pyhanabi_move_t* move);
bool ObsCardPlayableOnFireworks(const pyhanabi_observation_t* observation,
                                int color, int rank);

/* ObservationEncoder functions. */
void NewObservationEncoder(pyhanabi_observation_encoder_t* encoder,
                           pyhanabi_game_t* game, int type);
void DeleteObservationEncoder(pyhanabi_observation_encoder_t* encoder);
char* ObservationShape(pyhanabi_observation_encoder_t* encoder);
char* OwnHandShape(pyhanabi_observation_encoder_t* encoder);
char* EncodeObservation(pyhanabi_observation_encoder_t* encoder,
                        pyhanabi_observation_t* observation);
char* EncodeOwnHandObservation(pyhanabi_observation_encoder_t* encoder,
                        pyhanabi_observation_t* observation);

} /* extern "C" */

#endif
