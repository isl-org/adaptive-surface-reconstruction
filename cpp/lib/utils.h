//
// Copyright 2021 Intel (Autonomous Agents Lab)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
#pragma once

#include <string>

#include "asr.hpp"
#include "asr_config.h"

ASR_NAMESPACE_BEGIN

/// Primary print function for the library
void Print(const std::string& msg, int level);

/// Like Print but prepends the function name
#define PrintFN(msg, level) \
    Print(std::string(__PRETTY_FUNCTION__) + ": " + msg, level)

ASR_NAMESPACE_END
