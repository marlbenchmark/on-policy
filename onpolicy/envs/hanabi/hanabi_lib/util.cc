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

#include "util.h"

#include <string>

namespace hanabi_learning_env {

char ColorIndexToChar(int color) {
  if (color >= 0 && color <= kMaxNumColors) {
    return "RYGWB"[color];
  } else {
    return 'X';
  }
}

char RankIndexToChar(int rank) {
  if (rank >= 0 && rank <= kMaxNumRanks) {
    return "12345"[rank];
  } else {
    return 'X';
  }
}

template <>
int ParameterValue<int>(
    const std::unordered_map<std::string, std::string>& params,
    const std::string& key, int default_value) {
  auto iter = params.find(key);
  if (iter == params.end()) {
    return default_value;
  }

  return std::stoi(iter->second);
}

template <>
std::string ParameterValue<std::string>(
    const std::unordered_map<std::string, std::string>& params,
    const std::string& key, std::string default_value) {
  auto iter = params.find(key);
  if (iter == params.end()) {
    return default_value;
  }

  return iter->second;
}

template <>
double ParameterValue<double>(
    const std::unordered_map<std::string, std::string>& params,
    const std::string& key, double default_value) {
  auto iter = params.find(key);
  if (iter == params.end()) {
    return default_value;
  }

  return std::stod(iter->second);
}

template <>
bool ParameterValue<bool>(
    const std::unordered_map<std::string, std::string>& params,
    const std::string& key, bool default_value) {
  auto iter = params.find(key);
  if (iter == params.end()) {
    return default_value;
  }

  return (iter->second == "1" || iter->second == "true" ||
                  iter->second == "True"
              ? true
              : false);
}

}  // namespace hanabi_learning_env
