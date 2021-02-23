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

#ifndef __HANABI_CARD_H__
#define __HANABI_CARD_H__

#include <string>

namespace hanabi_learning_env {

class HanabiCard {
 public:
  HanabiCard(int color, int rank) : color_(color), rank_(rank) {}
  HanabiCard() = default;  // Create an invalid card.
  bool operator==(const HanabiCard& other_card) const;
  bool IsValid() const { return color_ >= 0 && rank_ >= 0; }
  std::string ToString() const;
  int Color() const { return color_; }
  int Rank() const { return rank_; }

 private:
  int color_ = -1;  // 0 indexed card color.
  int rank_ = -1;   // 0 indexed card rank.
};

}  // namespace hanabi_learning_env

#endif
