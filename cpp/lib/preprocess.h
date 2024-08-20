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
#include <vector>
#include <stddef.h>

#include "asr_config.h"

ASR_NAMESPACE_BEGIN

template <class T>
void FilterVector(std::vector<T>& vec,
                  const std::vector<char>& mask,
                  const int element_size = 1) {
    size_t idx = 0;
    for (size_t i = 0; i < mask.size(); ++i) {
        if (mask[i]) {
            for (int j = 0; j < element_size; ++j) {
                vec[idx] = vec[element_size * i + j];
                ++idx;
            }
        }
    }
    vec.resize(idx);
}

void ComputeInlierAndRadii(std::vector<char>& inlier,
                           std::vector<float>& radii,
                           const float* const points,
                           const size_t num_points,
                           const int k,
                           const float radius_fraction,
                           const int outlier_threshold);

void ComputeInlierFromDensity(std::vector<char>& inlier,
                              const float* const points,
                              const float* const radii,
                              const size_t num_points,
                              const double density_percentile_threshold);

ASR_NAMESPACE_END