//
// Copyright 2022 Intel (Autonomous Agents Lab)
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
#include "asr.hpp"

#include <dlfcn.h>
#include <open3d/utility/FileSystem.h>
#include <torch/script.h>

#include <stdexcept>

#include "contouring.h"
#include "grid.h"
#include "nsearch.h"
#include "octree.h"
#include "postprocess.h"
#include "preprocess.h"
#include "utils.h"

ASR_NAMESPACE_BEGIN

std::function<void(const char*)>& GetPrintCallbackFunction(int level) {
    static std::array<std::function<void(const char*)>, 4> callbacks;
    return callbacks[level];
}

void SetPrintCallbackFunction(std::function<void(const char*)> print_callback,
                              std::vector<int> levels) {
    for (int level : levels) {
        if (level < DEBUG || level > ERROR) {
            throw std::runtime_error("invalid verbosity level");
        }
        GetPrintCallbackFunction(level) = print_callback;
    }
}

std::string InitResourceDir() {
    if (const char* ptr = std::getenv("ASR_RESOURCE_DIR")) {
        return ptr;
    }

    Dl_info info;
    if (!dladdr((const void*)InitResourceDir, &info)) {
        throw std::runtime_error("dladdr failed");
    }
    std::string result = open3d::utility::filesystem::GetFileParentDirectory(
                                 info.dli_fname) +
                         "/asr_resources/";
    return result;
}

std::string& _RESOURCE_DIR() {
    static std::string dir = InitResourceDir();
    return dir;
}

void SetResourceDir(const char* path) { _RESOURCE_DIR() = path; }

const char* GetResourceDir() { return _RESOURCE_DIR().c_str(); }

const char* GetVersionStr() { return ASR_VERSION; }

const char* GetThirdPartyNotices() {
    // clang-format off
    return
    #include "third_party_notices.inl"
    ;
    // clang-format on
}

ReconstructSurfaceParams InitReconstructSurfaceParams() {
    ReconstructSurfaceParams result;
    result.point_radius_scale = 1.f;
    result.point_radius_estimation_knn = 24;
    result.density_percentile_threshold = 10.f;
    result.octree_max_depth = 21;
    result.contouring_value_threshold = 1.f;
    result.keep_n_connected_components = std::numeric_limits<int64_t>::max();
    result.minimum_component_size = 3;
    return result;
}

std::shared_ptr<ASRTriangleMesh> ReconstructSurface(
        std::vector<float>& points,
        std::vector<float>& normals,
        std::vector<float>& radii,
        const ReconstructSurfaceParams& params) {
    InitResourceDir();
    if (points.empty()) {
        throw std::runtime_error("points is null!\n");
    }

    if (points.size() != normals.size()) {
        throw std::runtime_error(
                "points vector does not have the same size as the normals "
                "vector.");
    }
    if (points.size() % 3) {
        throw std::runtime_error("points vector size is not a multiple of 3.");
    }

    size_t num_points = points.size() / 3;
    // preprocess
    {
        Print("preprocessing\n", INFO);
        std::vector<char> inlier_mask;
        if (radii.size()) {
            if (radii.size() != num_points) {
                throw std::runtime_error(
                        "number of radii differs from the number of points.");
            }
            ComputeInlierFromDensity(inlier_mask, points.data(), radii.data(),
                                     num_points,
                                     params.density_percentile_threshold);
        } else {
            // estimate radius
            ComputeInlierAndRadii(inlier_mask, radii, points.data(), num_points,
                                  params.point_radius_estimation_knn, 0.5, 1);
        }
        FilterVector(points, inlier_mask, 3);
        FilterVector(normals, inlier_mask, 3);
        FilterVector(radii, inlier_mask);
    }
    num_points = points.size() / 3;

    std::string model_path = GetResourceDir() + std::string("/model.pt");
    torch::jit::script::Module model = torch::jit::load(model_path);
    for (const auto& method : model.get_methods())
        std::cout << method.name() << "\n";

    // create the grid hierarchy
    Print("grid building\n", INFO);
    std::vector<std::shared_ptr<ASRGrid>> grids;
    std::vector<size_t> dual_vertex_indices;  // needed for contouring later
    {
        Eigen::Map<const MatXf> points_(points.data(), 3, num_points);
        Vec3f bb_min = points_.rowwise().minCoeff();
        Vec3f bb_max = points_.rowwise().maxCoeff();
        auto tree = CreateOctreeFromPoints(
                points.data(), num_points, radii.data(), bb_min, bb_max,
                params.point_radius_scale, 0, params.octree_max_depth);
        CreateDualVertexIndices(dual_vertex_indices, *tree);

        grids = CreateGridsFromOctree(*tree, 5, true);
    }

    torch::Dict<std::string, torch::Tensor> input_dict;
    {
        auto options = torch::TensorOptions()
                               .dtype(torch::kFloat32)
                               .device(torch::kCPU);
        auto points_torch = torch::from_blob(points.data(),
                                             {ssize_t(num_points), 3}, options);
        input_dict.insert("points", points_torch);

        torch::Tensor feats = torch::empty({ssize_t(num_points), 4}, options);
        float* ptr = feats.data_ptr<float>();
        for (size_t i = 0; i < num_points; i++, ptr += 4) {
            ptr[0] = normals[i * 3 + 0];
            ptr[1] = normals[i * 3 + 1];
            ptr[2] = normals[i * 3 + 2];
            ptr[3] = 1.f;
        }
        input_dict.insert("feats", feats);
    }

    for (size_t i = 0; i < grids.size(); ++i) {
        auto& g = grids[i];
        if (g->voxel_centers.size()) {
            std::string key = "voxel_centers" + std::to_string(i);
            auto options = torch::TensorOptions()
                                   .dtype(torch::kFloat32)
                                   .device(torch::kCPU);
            auto tensor = torch::from_blob(
                    g->voxel_centers.data(),
                    {ssize_t(g->voxel_centers.size() / 3), 3}, options);
            input_dict.insert(key, tensor);
        }
        if (g->voxel_sizes.size()) {
            std::string key = "voxel_sizes" + std::to_string(i);
            auto options = torch::TensorOptions()
                                   .dtype(torch::kFloat32)
                                   .device(torch::kCPU);
            auto tensor =
                    torch::from_blob(g->voxel_sizes.data(),
                                     {ssize_t(g->voxel_sizes.size())}, options);
            input_dict.insert(key, tensor);
        }
        if (g->neighbors_index.size()) {
            std::string key = "neighbors_index" + std::to_string(i);
            auto options = torch::TensorOptions()
                                   .dtype(torch::kInt32)
                                   .device(torch::kCPU);
            auto tensor = torch::from_blob(g->neighbors_index.data(),
                                           {ssize_t(g->neighbors_index.size())},
                                           options);
            input_dict.insert(key, tensor);
        }
        if (g->neighbors_kernel_index.size()) {
            std::string key = "neighbors_kernel_index" + std::to_string(i);
            auto options = torch::TensorOptions()
                                   .dtype(torch::kUInt8)
                                   .device(torch::kCPU);
            auto tensor = torch::from_blob(
                    g->neighbors_kernel_index.data(),
                    {ssize_t(g->neighbors_kernel_index.size())}, options);
            input_dict.insert(key, tensor);
        }
        if (g->neighbors_row_splits.size()) {
            std::string key = "neighbors_row_splits" + std::to_string(i);
            auto options = torch::TensorOptions()
                                   .dtype(torch::kInt64)
                                   .device(torch::kCPU);
            auto tensor = torch::from_blob(
                    g->neighbors_row_splits.data(),
                    {ssize_t(g->neighbors_row_splits.size())}, options);
            input_dict.insert(key, tensor);
        }
        if (g->up_neighbors_index.size()) {
            std::string key = "up_neighbors_index" + std::to_string(i);
            auto options = torch::TensorOptions()
                                   .dtype(torch::kInt32)
                                   .device(torch::kCPU);
            auto tensor = torch::from_blob(
                    g->up_neighbors_index.data(),
                    {ssize_t(g->up_neighbors_index.size())}, options);
            input_dict.insert(key, tensor);
        }
        if (g->up_neighbors_kernel_index.size()) {
            std::string key = "up_neighbors_kernel_index" + std::to_string(i);
            auto options = torch::TensorOptions()
                                   .dtype(torch::kUInt8)
                                   .device(torch::kCPU);
            auto tensor = torch::from_blob(
                    g->up_neighbors_kernel_index.data(),
                    {ssize_t(g->up_neighbors_kernel_index.size())}, options);
            input_dict.insert(key, tensor);
        }
        if (g->up_neighbors_row_splits.size()) {
            std::string key = "up_neighbors_row_splits" + std::to_string(i);
            auto options = torch::TensorOptions()
                                   .dtype(torch::kInt64)
                                   .device(torch::kCPU);
            auto tensor = torch::from_blob(
                    g->up_neighbors_row_splits.data(),
                    {ssize_t(g->up_neighbors_row_splits.size())}, options);
            input_dict.insert(key, tensor);
        }
    }

    // aggregation
    Print("aggregate\n", INFO);

    std::vector<int32_t> aggregation_neighbors_index;
    std::vector<float> aggregation_neighbors_dist;
    std::vector<int64_t> aggregation_row_splits;
    std::vector<float> aggregation_scale_compat;
    ComputeAggregationNeighborsAndScaleCompatibility(
            aggregation_neighbors_index, aggregation_neighbors_dist,
            aggregation_row_splits, aggregation_scale_compat, points, radii,
            grids[0]->voxel_centers, grids[0]->voxel_sizes);
    {
        std::string key = "aggregation_neighbors_index";
        auto options =
                torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
        auto tensor = torch::from_blob(
                aggregation_neighbors_index.data(),
                {ssize_t(aggregation_neighbors_index.size())}, options);
        input_dict.insert(key, tensor);
    }
    {
        std::string key = "aggregation_neighbors_dist";
        auto options = torch::TensorOptions()
                               .dtype(torch::kFloat32)
                               .device(torch::kCPU);
        auto tensor = torch::from_blob(
                aggregation_neighbors_dist.data(),
                {ssize_t(aggregation_neighbors_dist.size())}, options);
        input_dict.insert(key, tensor);
    }
    {
        std::string key = "aggregation_row_splits";
        auto options =
                torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
        auto tensor = torch::from_blob(aggregation_row_splits.data(),
                                       {ssize_t(aggregation_row_splits.size())},
                                       options);
        input_dict.insert(key, tensor);
    }

    {
        std::string key = "aggregation_scale_compat";
        auto options = torch::TensorOptions()
                               .dtype(torch::kFloat32)
                               .device(torch::kCPU);
        auto tensor = torch::from_blob(
                aggregation_scale_compat.data(),
                {ssize_t(aggregation_scale_compat.size())}, options);
        input_dict.insert(key, tensor);
    }

    Print("network aggregate\n", INFO);
    auto feats1_and_importance = model.run_method("aggregate", input_dict);

    // network

    Print("network unet\n", INFO);
    auto code = model.run_method("unet", feats1_and_importance, input_dict);

    // decode
    Print("network decode\n", INFO);
    auto shift = torch::zeros({code.toTensor().size(0), 3}, torch::kFloat32);
    auto values =
            model.run_method("decode", shift, code).toTensor().contiguous();
    // values has a signed and unsigned distance value and the shape is {N,2}

    // contouring
    std::vector<float> values_vec(values.numel());
    std::memcpy(values_vec.data(), values.data_ptr(),
                sizeof(float) * values_vec.size());
    // multiply with the voxel size to get the signed distance value.
    for (size_t i = 0; i < grids[0]->voxel_sizes.size(); ++i) {
        values_vec[i * 2] *= grids[0]->voxel_sizes[i];
    }

    auto result = std::make_shared<ASRTriangleMesh>();

    CreateTriangleMesh(result->vertices, result->triangles, values_vec,
                       dual_vertex_indices, grids[0]->voxel_centers,
                       params.contouring_value_threshold);

    RemoveConnectedComponents(result->vertices, result->triangles,
                              params.keep_n_connected_components,
                              params.minimum_component_size);

    return result;
}

ASR_NAMESPACE_END
