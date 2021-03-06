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

#if defined(__cplusplus)
#define ASR_API_EXTERN_C extern "C"
#define ASR_API_EXTERN_C_BEGIN ASR_API_EXTERN_C {
#define ASR_API_EXTERN_C_END }
#define ASR_NAMESPACE_BEGIN namespace asr {
#define ASR_NAMESPACE_END }
#else
#define ASR_API_EXTERN_C
#define ASR_API_EXTERN_C_BEGIN
#define ASR_API_EXTERN_C_END
#endif

#include "asr_config_generated.h"