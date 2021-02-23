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

// The standard Open Hanabi observation encoders. These encoders translate
// HanabiObservations to input tensors that an agent can train on.

#ifndef __CANONICAL_ENCODERS_H__
#define __CANONICAL_ENCODERS_H__

#include <vector>

#include "hanabi_game.h"
#include "hanabi_observation.h"
#include "observation_encoder.h"

namespace hanabi_learning_env {

// This is the canonical observation encoding.
class CanonicalObservationEncoder : public ObservationEncoder {
 public:
  explicit CanonicalObservationEncoder(const HanabiGame* parent_game)
      : parent_game_(parent_game) {}

  std::vector<int> Shape() const override;
  std::vector<int> OwnHandShape() const override;
  std::vector<int> Encode(const HanabiObservation& obs) const override;
  std::vector<int> EncodeOwnHand(const HanabiObservation& obs) const override;

  ObservationEncoder::Type type() const override {
    return ObservationEncoder::Type::kCanonical;
  }

 private:
  const HanabiGame* parent_game_ = nullptr;
};

}  // namespace hanabi_learning_env

#endif
