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
#include "preprocess.h"

#include <iostream>

#include "nsearch.h"
#include "utils.h"

ASR_NAMESPACE_BEGIN

void ComputeInlierAndRadii(std::vector<char>& inlier,
                           std::vector<float>& radii,
                           const float* const points,
                           const size_t num_points,
                           const int k,
                           const float radius_fraction,
                           const int outlier_threshold) {
    PrintFN("build index\n", DEBUG);
    KDTree kdtree(points, num_points);
    PrintFN("compute k radius\n", DEBUG);
    radii = kdtree.ComputeKRadius(k);
    PrintFN("compute inlier\n", DEBUG);
    inlier = kdtree.ComputeInlier(radii.data(), radius_fraction, k,
                                  outlier_threshold);
}

void ComputeInlierFromDensity(std::vector<char>& inlier,
                              const float* const points,
                              const float* const radii,
                              const size_t num_points,
                              const double density_percentile_threshold) {
    PrintFN("build index\n", DEBUG);
    KDTree kdtree(points, num_points);
    auto radius_neighbors = kdtree.ComputeRadiusNeighbors(radii);

    size_t middle = (density_percentile_threshold / 100) * num_points;
    middle = std::min(num_points, std::max<size_t>(1, middle));
    PrintFN("partial sort\n", DEBUG);
    std::partial_sort(radius_neighbors.begin(),
                      radius_neighbors.begin() + middle,
                      radius_neighbors.end());
    int threshold = radius_neighbors[middle - 1];

    inlier.resize(num_points);
    for (size_t i = 0; i < num_points; ++i) {
        inlier[i] = radius_neighbors[i] > threshold;
    }
}

ASR_NAMESPACE_END