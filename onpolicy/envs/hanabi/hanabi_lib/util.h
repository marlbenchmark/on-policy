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

#ifndef __UTIL_H__
#define __UTIL_H__

#include <cassert>
#include <cstdlib>
#include <string>
#include <unordered_map>

namespace hanabi_learning_env {

constexpr int kMaxNumColors = 5;
constexpr int kMaxNumRanks = 5;

// Returns a character representation of an integer color/rank index.
char ColorIndexToChar(int color);
char RankIndexToChar(int rank);

// Returns string associated with key in params, parsed as template type.
// If key is not in params, returns the provided default value.
template <class T>
T ParameterValue(const std::unordered_map<std::string, std::string>& params,
                 const std::string& key, T default_value);

template <>
int ParameterValue(const std::unordered_map<std::string, std::string>& params,
                   const std::string& key, int default_value);
template <>
double ParameterValue(
    const std::unordered_map<std::string, std::string>& params,
    const std::string& key, double default_value);
template <>
std::string ParameterValue(
    const std::unordered_map<std::string, std::string>& params,
    const std::string& key, std::string default_value);
template <>
bool ParameterValue(const std::unordered_map<std::string, std::string>& params,
                    const std::string& key, bool default_value);

#if defined(NDEBUG)
#define REQUIRE(expr)                                                        \
  (expr ? (void)0                                                            \
        : (fprintf(stderr, "Input requirements failed at %s:%d in %s: %s\n", \
                   __FILE__, __LINE__, __func__, #expr),                     \
           std::abort()))
#else
#define REQUIRE(expr) assert(expr)
#endif

}  // namespace hanabi_learning_env

#endif
