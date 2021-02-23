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

// A helper object to translate Hanabi observations to agent inputs
// (e.g. tensors).

#ifndef __OBSERVATION_ENCODER_H__
#define __OBSERVATION_ENCODER_H__

#include <vector>

#include "hanabi_observation.h"

namespace hanabi_learning_env {

class ObservationEncoder {
 public:
  enum Type { kCanonical = 0 };
  virtual ~ObservationEncoder() = default;

  // Returns the shape (dimension sizes of the tensor).
  virtual std::vector<int> Shape() const = 0;
  virtual std::vector<int> OwnHandShape() const = 0;

  // All of the canonical observation encodings are vectors of bits. We can
  // change this if we want something more general (e.g. floats or doubles).
  virtual std::vector<int> Encode(const HanabiObservation& obs) const = 0;
  virtual std::vector<int> EncodeOwnHand(const HanabiObservation& obs) const = 0;

  // Return the type of this encoder.
  virtual Type type() const = 0;
};

}  // namespace hanabi_learning_env

#endif
