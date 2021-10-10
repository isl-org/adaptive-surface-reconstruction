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

#include <functional>
#include <memory>

#include "asr_config.h"
#include "asr_types.h"

ASR_NAMESPACE_BEGIN

/// Verbosity levels used for setting up the print function callbacks.
enum VERBOSITY_LEVELS { DEBUG = 0, INFO = 1, WARN = 2, ERROR = 3 };

/// Sets the print callback for specific verbosity levels.
/// \param print_callback Callback function used for printing.
/// \param levels The verbosity levelsfor which the callback should be
/// installed.
void SetPrintCallbackFunction(
        std::function<void(const std::string&)> print_callback,
        std::vector<int> levels);

/// Sets the resource dir to the \p path .
void SetResourceDir(const std::string& path);

/// Returns the resource dir. The initial default resource dir is
/// <tt>$ORIGIN/asr_resources</tt>
std::string GetResourceDir();

/// Returns the version string of the library.
std::string GetVersionStr();

/// Returns the third-party notices as string.
std::string GetThirdPartyNotices();

/// Struct with the parameters for the ReconstructSurface function.
/// Use InitReconstructSurfaceParams() to create instances.
struct ReconstructSurfaceParams {
    /// Scalar used for scaling the per point radius/footprint size.
    float point_radius_scale;

    /// The density threshold as percentile. To remove 10% of the points with
    /// the lowest density set this to 10.
    float density_percentile_threshold;

    /// The number of neighbors to use for estimating the point radius/footprint
    /// size. This parameter is used if the radii array is empty.
    int point_radius_estimation_knn;

    /// The maximum depth for the octree. This value should be in the range
    /// [6..21].
    int octree_max_depth;

    /// The threshold for the unsigned distance for generating mesh vertices.
    /// For small values only vertices will be generated where the algorithm is
    /// certain about the surface. For large values the algorithm will generate
    /// vertices even far away from the input data. The range of this value is
    /// [0..2] and the default value is 1.
    float contouring_value_threshold;

    /// The number of connected components to keep in the postprocessing.
    /// Set this to \w n to keep only the \w n largest connected components.
    /// By default all components will be kept.
    int64_t keep_n_connected_components;

    /// The minimum size of a component to be kept. The default is 3 which means
    /// even single triangles will be kept.
    int64_t minimum_component_size;
};

/// Returns a ReconstructSurfaceParams struct with default values.
ReconstructSurfaceParams InitReconstructSurfaceParams();

/// Reconstruct the surface from an oriented point cloud with optional
/// radii/scale/footprint information.
///
/// \param points Points with shape [num_points, 3].
/// \param normals Normals with shape [num_points, 3].
/// \param radii The footprint size for each point with shape [num_points] or
///        an empty vector.
/// \param params Struct with parameters. Use InitReconstructSurfaceParams() to
///        create a set of default parameters.
/// \return A triangle mesh.
std::shared_ptr<ASRTriangleMesh> ReconstructSurface(
        std::vector<float>& points,
        std::vector<float>& normals,
        std::vector<float>& radii,
        const ReconstructSurfaceParams& params);

ASR_NAMESPACE_END
