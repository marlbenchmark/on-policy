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

#include "hanabi_move.h"

#include "util.h"

namespace hanabi_learning_env {

bool HanabiMove::operator==(const HanabiMove& other_move) const {
  if (MoveType() != other_move.MoveType()) {
    return false;
  }
  switch (MoveType()) {
    case kPlay:
    case kDiscard:
      return CardIndex() == other_move.CardIndex();
    case kRevealColor:
      return TargetOffset() == other_move.TargetOffset() &&
             Color() == other_move.Color();
    case kRevealRank:
      return TargetOffset() == other_move.TargetOffset() &&
             Rank() == other_move.Rank();
    case kDeal:
      return Color() == other_move.Color() && Rank() == other_move.Rank();
    default:
      return true;
  }
}

std::string HanabiMove::ToString() const {
  switch (MoveType()) {
    case kPlay:
      return "(Play " + std::to_string(CardIndex()) + ")";
    case kDiscard:
      return "(Discard " + std::to_string(CardIndex()) + ")";
    case kRevealColor:
      return "(Reveal player +" + std::to_string(TargetOffset()) + " color " +
             ColorIndexToChar(Color()) + ")";
    case kRevealRank:
      return "(Reveal player +" + std::to_string(TargetOffset()) + " rank " +
             RankIndexToChar(Rank()) + ")";
    case kDeal:
      if (color_ >= 0) {
        return std::string("(Deal ") + ColorIndexToChar(Color()) +
               RankIndexToChar(Rank()) + ")";
      } else {
        return std::string("(Deal XX)");
      }
    default:
      return "(INVALID)";
  }
}

}  // namespace hanabi_learning_env
