# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Clean-up script to bring the repository back to a pre-cmake state.

#!/bin/sh
if [ -f Makefile ]
then
  make clean
fi

rm -rf *.pyc agents/*.pyc __pycache__ agents/__pycache__ CMakeCache.txt CMakeFiles Makefile cmake_install.cmake  hanabi_lib/CMakeFiles hanabi_lib/Makefile hanabi_lib/cmake_install.cmake
