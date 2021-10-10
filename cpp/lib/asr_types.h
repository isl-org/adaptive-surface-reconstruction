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

/// Class describing a triangle mesh.
struct ASRTriangleMesh {
    /// Vertices with shape [num_vertices,3] as a flat vector.
    std::vector<float> vertices;
    /// Triangles with shape [num_triangles,3] as a flat vector.
    std::vector<int32_t> triangles;
};

/// Class representing a grid and its connectivity information.
struct ASRGrid {
    /// The location codes for each voxel with shape [num_voxels].
    std::vector<uint64_t> voxel_keys;

    /// The xyz position of each voxel center with shape [num_voxels,3] as a
    /// flat vector.
    std::vector<float> voxel_centers;

    /// The size (edge length) of each voxel with shape [num_voxels].
    std::vector<float> voxel_sizes;

    /// Stores the indices of the face-adjacent voxel neighbors.
    /// The start and end position for each voxel is defined by the array
    /// neighbors_row_splits .
    std::vector<int32_t> neighbors_index;

    /// The index for the kernel element to use in the
    /// generalized sparse conv. Start and end for each voxel (row) is defined
    /// by the neighbors_row_splits array.
    std::vector<uint8_t> neighbors_kernel_index;

    /// Defines the start and end for each voxel (row) for the 'neighbors_index'
    /// and 'neighbors_kernel_index' arrays.
    std::vector<int64_t> neighbors_row_splits;

    /// Same as neighbors_index but defines the connectivity
    /// from the current grid to the next finer grid.
    std::vector<int32_t> up_neighbors_index;

    /// Same as neighbors_kernel_index but defines
    /// the kernel element to use for the transition to the next finer grid.
    std::vector<uint8_t> up_neighbors_kernel_index;

    /// Same as neighbors_row_splits but for the
    /// up_neighbors_index and up_neighbors_kernel_index arrays.
    std::vector<int64_t> up_neighbors_row_splits;
};