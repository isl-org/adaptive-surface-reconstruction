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
#include "nsearch.h"

#include <open3d/core/nns/NearestNeighborSearch.h>
#include <tbb/parallel_for.h>

ASR_NAMESPACE_BEGIN

KDTree::KDTree(const float* const points, size_t num_points)
    : points_(points), num_points_(num_points) {
    adaptor_ = std::make_unique<Adaptor<float>>(num_points, points);
    index_ = std::make_unique<KDIndex_t<float>>(3, *adaptor_);
    index_->buildIndex();
}

std::vector<float> KDTree::ComputeKRadius(const int k) {
    std::vector<float> result(num_points_);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, num_points_),
                      [&](const tbb::blocked_range<size_t>& r) {
                          std::vector<size_t> result_indices(k);
                          std::vector<float> result_distances(k);
                          for (size_t i = r.begin(); i != r.end(); ++i) {
                              int num_valid = index_->knnSearch(
                                      points_ + 3 * i, k, result_indices.data(),
                                      result_distances.data());

                              float radius = 0;
                              for (int n = 0; n < num_valid; ++n) {
                                  if (result_distances[n] > radius) {
                                      radius = result_distances[n];
                                  }
                              }
                              result[i] = std::sqrt(radius);
                          }
                      });
    return result;
}

std::vector<char> KDTree::ComputeInlier(const float* const radii,
                                        const float radius_fraction,
                                        const int k,
                                        const int outlier_threshold) {
    std::vector<char> result(num_points_);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, num_points_),
                      [&](const tbb::blocked_range<size_t>& r) {
                          std::vector<size_t> result_indices(k);
                          std::vector<float> result_distances(k);
                          for (size_t i = r.begin(); i != r.end(); ++i) {
                              int num_valid = index_->knnSearch(
                                      points_ + 3 * i, k, result_indices.data(),
                                      result_distances.data());

                              // we count the neighbors that have a smaller
                              // radius than a fraction of the current points
                              // radius.
                              float radius = radii[i] * radius_fraction;
                              int outlier_votes = 0;
                              for (int n = 0; n < num_valid &&
                                              outlier_votes < outlier_threshold;
                                   ++n) {
                                  if (radii[result_indices[n]] < radius) {
                                      ++outlier_votes;
                                  }
                              }
                              result[i] = outlier_votes < outlier_threshold;
                          }
                      });

    return result;
}

std::vector<int> KDTree::ComputeRadiusNeighbors(const float* const radii) {
    nanoflann::SearchParams search_params(32, 0, false);

    std::vector<int> result(num_points_);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, num_points_),
                      [&](const tbb::blocked_range<size_t>& r) {
                          std::vector<std::pair<size_t, float>> search_result;
                          for (size_t i = r.begin(); i != r.end(); ++i) {
                              int num_valid = index_->radiusSearch(
                                      points_ + 3 * i, radii[i] * radii[i],
                                      search_result, search_params);

                              result[i] = num_valid;
                          }
                      });
    return result;
}

void ComputeAggregationNeighborsAndScaleCompatibility(
        std::vector<int32_t>& neighbors_index,
        std::vector<float>& neighbors_dist,
        std::vector<int64_t>& neighbors_row_splits,
        std::vector<float>& neighbors_scale_compat,
        const std::vector<float>& points,
        const std::vector<float>& point_radii,
        const std::vector<float>& voxel_centers,
        const std::vector<float>& voxel_sizes) {
    using namespace open3d::core;
    using namespace open3d::core::nns;

    auto DummyDeleter = [](void*) { /*empty*/ };
    auto CreateTensorView = [&](void* data, SizeVector shape, Dtype dtype) {
        auto blob = std::make_shared<Blob>(Device("CPU:0"), data, DummyDeleter);
        Tensor tensor(shape, shape_util::DefaultStrides(shape), data, dtype,
                      blob);
        return std::make_tuple(tensor, blob);
    };

    auto dataset =
            CreateTensorView(const_cast<float*>(points.data()),
                             {int64_t(point_radii.size()), 3}, Dtype::Float32);
    NearestNeighborSearch nns(std::get<0>(dataset));
    nns.MultiRadiusIndex();

    auto query_points =
            CreateTensorView(const_cast<float*>(voxel_centers.data()),
                             {int64_t(voxel_sizes.size()), 3}, Dtype::Float32);
    auto query_radii =
            CreateTensorView(const_cast<float*>(voxel_sizes.data()),
                             {int64_t(voxel_sizes.size())}, Dtype::Float32);
    {
        auto ans = nns.MultiRadiusSearch(std::get<0>(query_points),
                                         std::get<0>(query_radii));
        neighbors_index =
                std::get<0>(ans).To(Dtype::Int32).ToFlatVector<int32_t>();
        neighbors_dist = std::get<1>(ans).ToFlatVector<float>();
        neighbors_row_splits = std::get<2>(ans).ToFlatVector<int64_t>();
    }

    // compute scale compatibility
    float gamma = 2;
    neighbors_scale_compat.resize(neighbors_index.size());

    for (size_t i = 0; i < neighbors_row_splits.size() - 1; ++i) {
        float size_a = voxel_sizes[i];
        int64_t n_begin = neighbors_row_splits[i];
        int64_t n_end = neighbors_row_splits[i + 1];
        for (int64_t j = n_begin; j < n_end; ++j) {
            float size_b = 2 * point_radii[neighbors_index[j]];
            neighbors_scale_compat[j] = std::pow(
                    std::min(size_a, size_b) / std::max(size_a, size_b), gamma);
        }
    }
}

ASR_NAMESPACE_END
