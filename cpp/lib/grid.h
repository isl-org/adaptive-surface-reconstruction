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

#include "asr_types.h"
#include "octree.h"

ASR_NAMESPACE_BEGIN

std::vector<std::shared_ptr<ASRGrid>> CreateGridsFromOctree(
        const Octree& tree,
        const int num_levels,
        const bool voxel_info_all_levels);

void CreateDualVertexIndices(std::vector<size_t>& dual_vertex_indices,
                             const Octree& tree);

ASR_NAMESPACE_END