//
// Copyright 2024 Intel
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

#include "utils.h"

#include "asr.hpp"

ASR_NAMESPACE_BEGIN
std::function<void(const char*)>& GetPrintCallbackFunction(int level);

void Print(const std::string& msg, int level) {
    if (GetPrintCallbackFunction(level))
        GetPrintCallbackFunction(level)(msg.c_str());
}

ASR_NAMESPACE_END
