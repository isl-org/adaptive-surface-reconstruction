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

#include <cstdint>
#include <vector>

#include "asr_config.h"

ASR_NAMESPACE_BEGIN

void ConnectedComponents(std::vector<int64_t>& component,
                         std::vector<int64_t>& component_sizes,
                         const std::vector<float>& vertices,
                         std::vector<int32_t>& triangles);

void RemoveConnectedComponents(std::vector<float>& vertices,
                               std::vector<int32_t>& triangles,
                               int64_t keep_n_largest_components,
                               int64_t minimum_component_size);

void FilterTriangleMeshVertices(std::vector<float>& vertices,
                                std::vector<int32_t>& triangles,
                                const std::vector<int8_t>& mask_vertices);

ASR_NAMESPACE_END