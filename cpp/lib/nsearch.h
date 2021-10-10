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

#include <memory>

#include "asr_config.h"
#include "nanoflann.hpp"

ASR_NAMESPACE_BEGIN

/// Adaptor for nanoflann
template <class T>
class Adaptor {
public:
    Adaptor(size_t num_points, const T* const data)
        : num_points(num_points), data(data) {}

    inline size_t kdtree_get_point_count() const { return num_points; }

    inline T kdtree_get_pt(const size_t idx, int dim) const {
        return data[3 * idx + dim];
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const {
        return false;
    }

private:
    size_t num_points;
    const T* const data;
};

template <class T>
using L2Adaptor_t = nanoflann::L2_Adaptor<T, Adaptor<T>>;

template <class T>
using KDIndex_t =
        nanoflann::KDTreeSingleIndexAdaptor<L2Adaptor_t<T>, Adaptor<T>, 3>;

class KDTree {
public:
    KDTree(const float* const points, size_t num_points);
    std::vector<float> ComputeKRadius(const int k);
    std::vector<char> ComputeInlier(const float* const radii,
                                    const float radius_fraction,
                                    const int k,
                                    const int outlier_threshold);
    std::vector<int> ComputeRadiusNeighbors(const float* const radii);

private:
    std::unique_ptr<Adaptor<float>> adaptor_;
    std::unique_ptr<KDIndex_t<float>> index_;
    const float* const points_;
    const size_t num_points_;
};

void ComputeAggregationNeighborsAndScaleCompatibility(
        std::vector<int32_t>& neighbors_index,
        std::vector<float>& neighbors_dist,
        std::vector<int64_t>& neighbors_row_splits,
        std::vector<float>& neighbors_scale_compat,
        const std::vector<float>& points,
        const std::vector<float>& point_radii,
        const std::vector<float>& voxel_centers,
        const std::vector<float>& voxel_sizes);

ASR_NAMESPACE_END