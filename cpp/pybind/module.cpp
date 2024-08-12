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
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "asr.hpp"
#include "grid.h"
#include "nsearch.h"
#include "octree.h"
#include "postprocess.h"

namespace py = pybind11;
using namespace pybind11::literals;
using namespace asr;

namespace {
ReconstructSurfaceParams defaultReconstructSurfaceParams =
        InitReconstructSurfaceParams();

template <class T>
py::array_t<T> vector2pyarray_t(
        const std::vector<T>& data,
        std::vector<size_t> shape = std::vector<size_t>()) {
    if (shape.empty()) {
        shape.push_back(data.size());
    } else {
        size_t total = 1;
        for (size_t s : shape) {
            total *= s;
        }
        if (data.size() != total) {
            throw std::invalid_argument(
                    "cannot convert to py::array_t. shape is incompatible");
        }
    }
    py::array_t<T> arr(shape);
    std::memcpy(arr.mutable_data(0), data.data(), sizeof(T) * data.size());
    return arr;
}

}  // namespace

py::dict pyReconstructSurface(
        py::array_t<float, py::array::c_style | py::array::forcecast> points,
        py::array_t<float, py::array::c_style | py::array::forcecast> normals,
        py::array_t<float, py::array::c_style | py::array::forcecast> radii,
        float point_radius_scale,
        float density_percentile_threshold,
        int point_radius_estimation_knn,
        int octree_max_depth,
        float contouring_value_threshold,
        int64_t keep_n_connected_components,
        int64_t minimum_component_size) {
    if (points.ndim() != 2 || points.shape(1) != 3) {
        throw std::invalid_argument("points must have shape [num_points,3]");
    }
    if (normals.ndim() != 2 || normals.shape(0) != points.shape(0) ||
        normals.shape(1) != 3) {
        throw std::invalid_argument("normals must have shape [num_points,3]");
    }
    if (radii.size() > 0) {
        if (radii.ndim() != 1 || radii.shape(0) != points.shape(0)) {
            throw std::invalid_argument("radii must have shape [num_point3]");
        }
    }

    std::vector<float> points_(points.data(0), points.data(0) + points.size());
    std::vector<float> normals_(normals.data(0),
                                normals.data(0) + normals.size());
    std::vector<float> radii_(radii.data(0), radii.data(0) + radii.size());

    auto params = InitReconstructSurfaceParams();
    params.point_radius_scale = point_radius_scale;
    params.point_radius_estimation_knn = point_radius_estimation_knn;
    params.density_percentile_threshold = density_percentile_threshold;
    params.octree_max_depth = octree_max_depth;
    params.contouring_value_threshold = contouring_value_threshold;
    params.keep_n_connected_components = keep_n_connected_components;
    params.minimum_component_size = minimum_component_size;

    auto result = ReconstructSurface(points_, normals_, radii_, params);

    py::array_t<float> out_vertices = vector2pyarray_t(
            result->vertices, {result->vertices.size() / 3, 3});
    result->vertices.clear();
    result->vertices.shrink_to_fit();

    py::array_t<int32_t> out_triangles = vector2pyarray_t(
            result->triangles, {result->triangles.size() / 3, 3});
    result->triangles.clear();
    result->triangles.shrink_to_fit();

    return py::dict("vertices"_a = out_vertices, "triangles"_a = out_triangles);
}

py::dict pyRemoveConnectedComponents(
        py::array_t<float, py::array::c_style | py::array::forcecast> vertices,
        py::array_t<int32_t, py::array::c_style | py::array::forcecast>
                triangles,
        int64_t keep_n_largest_components,
        int64_t minimum_component_size) {
    if (vertices.ndim() != 2 || vertices.shape(1) != 3) {
        throw std::invalid_argument("vertices must have shape [N,3]");
    }
    if (triangles.ndim() != 2 || triangles.shape(1) != 3) {
        throw std::invalid_argument("triangles must have shape [N,3]");
    }

    std::vector<float> verts(vertices.data(0),
                             vertices.data(0) + vertices.size());
    std::vector<int32_t> tris(triangles.data(0),
                              triangles.data(0) + triangles.size());

    RemoveConnectedComponents(verts, tris, keep_n_largest_components,
                              minimum_component_size);

    py::array_t<float> out_verts =
            vector2pyarray_t(verts, {verts.size() / 3, 3});
    verts.clear();
    verts.shrink_to_fit();
    py::array_t<int32_t> out_tris({ssize_t(tris.size() / 3), ssize_t(3)},
                                  tris.data());
    tris.clear();
    tris.shrink_to_fit();

    return py::dict("vertices"_a = out_verts, "triangles"_a = out_tris);
}

std::shared_ptr<Octree> pyCreateOctreeFromPoints(
        py::array_t<float, py::array::c_style | py::array::forcecast> points,
        py::array_t<float, py::array::c_style | py::array::forcecast> radii,
        const Vec3f& bb_min,
        const Vec3f& bb_max,
        const float radius_scale,
        const int grow_steps,
        const int max_depth) {
    if (points.ndim() != 2 || points.shape(1) != 3) {
        throw std::invalid_argument("points must have shape [N,3]");
    }
    if (radii.ndim() != 1 || radii.shape(0) != points.shape(0)) {
        throw std::invalid_argument("radii must have shape [N]");
    }
    return CreateOctreeFromPoints(points.data(0), points.shape(0),
                                  radii.data(0), bb_min, bb_max, radius_scale,
                                  grow_steps, max_depth);
}

py::list pyCreateGridsFromOctree(std::shared_ptr<Octree> tree,
                                 const int num_levels,
                                 const bool voxel_info_all_levels) {
    auto grids =
            CreateGridsFromOctree(*tree, num_levels, voxel_info_all_levels);

    py::list result;

    for (auto& g : grids) {
        py::dict d;

        if (g->voxel_keys.size()) {
            d["voxel_keys"] = vector2pyarray_t(g->voxel_keys);
            g->voxel_keys.clear();
            g->voxel_keys.shrink_to_fit();
        }
        if (g->voxel_centers.size()) {
            d["voxel_centers"] = vector2pyarray_t(
                    g->voxel_centers, {g->voxel_centers.size() / 3, 3});
            g->voxel_centers.clear();
            g->voxel_centers.shrink_to_fit();
        }
        if (g->voxel_sizes.size()) {
            d["voxel_sizes"] = vector2pyarray_t(g->voxel_sizes);
            g->voxel_sizes.clear();
            g->voxel_sizes.shrink_to_fit();
        }
        if (g->neighbors_index.size()) {
            d["neighbors_index"] = vector2pyarray_t(g->neighbors_index);
            g->neighbors_index.clear();
            g->neighbors_index.shrink_to_fit();
        }
        if (g->neighbors_kernel_index.size()) {
            d["neighbors_kernel_index"] =
                    vector2pyarray_t(g->neighbors_kernel_index);
            g->neighbors_kernel_index.clear();
            g->neighbors_kernel_index.shrink_to_fit();
        }
        if (g->neighbors_row_splits.size()) {
            d["neighbors_row_splits"] =
                    vector2pyarray_t(g->neighbors_row_splits);
            g->neighbors_row_splits.clear();
            g->neighbors_row_splits.shrink_to_fit();
        }
        if (g->up_neighbors_index.size()) {
            d["up_neighbors_index"] = vector2pyarray_t(g->up_neighbors_index);
            g->up_neighbors_index.clear();
            g->up_neighbors_index.shrink_to_fit();
        }
        if (g->up_neighbors_kernel_index.size()) {
            d["up_neighbors_kernel_index"] =
                    vector2pyarray_t(g->up_neighbors_kernel_index);
            g->up_neighbors_kernel_index.clear();
            g->up_neighbors_kernel_index.shrink_to_fit();
        }
        if (g->up_neighbors_row_splits.size()) {
            d["up_neighbors_row_splits"] =
                    vector2pyarray_t(g->up_neighbors_row_splits);
            g->up_neighbors_row_splits.clear();
            g->up_neighbors_row_splits.shrink_to_fit();
        }
        result.append(d);
    }

    return result;
}

py::array_t<size_t> pyCreateDualVertexIndices(std::shared_ptr<Octree> tree) {
    std::vector<size_t> dual_indices;
    CreateDualVertexIndices(dual_indices, *tree);

    return vector2pyarray_t(dual_indices, {dual_indices.size() / 8, 8});
}

class pyKDTree {
public:
    pyKDTree(py::array_t<float, py::array::c_style> points) : points_(points) {
        if (points.ndim() != 2 || points.shape(1) != 3) {
            throw std::invalid_argument("points must have shape [N,3]");
        }
        tree_ = std::make_unique<KDTree>(points_.data(0), points_.shape(0));
    }

    py::array_t<float> ComputeKRadius(const int k) {
        auto tmp = tree_->ComputeKRadius(k);
        return vector2pyarray_t(tmp);
    }

    py::array_t<bool> ComputeInlier(
            py::array_t<float, py::array::c_style> radii,
            const float radius_fraction,
            const int k,
            const int outlier_threshold) {
        if (radii.ndim() != 1 || radii.shape(0) != points_.shape(0)) {
            throw std::invalid_argument("radii must have shape [num_points]");
        }

        auto inlier = tree_->ComputeInlier(radii.data(0), radius_fraction, k,
                                           outlier_threshold);
        py::array_t<bool> result({inlier.size()});
        for (size_t i = 0; i < inlier.size(); ++i) {
            result.mutable_at(i) = bool(inlier[i]);
        }
        return result;
    }

    std::vector<int> ComputeRadiusNeighbors(
            py::array_t<float, py::array::c_style> radii) {
        return tree_->ComputeRadiusNeighbors(radii.data(0));
    }

private:
    py::array_t<float, py::array::c_style> points_;
    std::unique_ptr<KDTree> tree_;
};

PYBIND11_MODULE(asrpybind, m) {
    m.doc() = "Python bindings for adaptive surface reconstruction";

    py::class_<Octree, std::shared_ptr<Octree>>(m, "Octree");

    m.def("get_version_str", &GetVersionStr, R"doc(Returns the version string.
)doc");

    m.def("get_third_party_notices", &GetThirdPartyNotices,
          R"doc(Returns the third-party notices.
)doc");

    m.def("reconstruct_surface", &pyReconstructSurface, "points"_a, "normals"_a,
          "radii"_a,
          "point_radius_scale"_a =
                  defaultReconstructSurfaceParams.point_radius_scale,
          "density_percentile_threshold"_a =
                  defaultReconstructSurfaceParams.density_percentile_threshold,
          "point_radius_estimation_knn"_a =
                  defaultReconstructSurfaceParams.point_radius_estimation_knn,
          "octree_max_depth"_a =
                  defaultReconstructSurfaceParams.octree_max_depth,
          "contouring_value_threshold"_a =
                  defaultReconstructSurfaceParams.contouring_value_threshold,
          "keep_n_connected_components"_a =
                  defaultReconstructSurfaceParams.keep_n_connected_components,
          "minimum_component_size"_a =
                  defaultReconstructSurfaceParams.minimum_component_size,
          R"doc(Reconstruct the surface from an oriented point cloud with optional radii/scale/footprint information.

Args:
    points (np.ndarray): Points with shape [num_points, 3] and dtype np.float32.

    normals (np.ndarray): Normals with shape [num_points, 3] and dtype np.float32.

    radii (np.ndarray): The footprint size for each point with shape [num_points]
        and dtype np.float32 or None.

    point_radius_scale (float): Scalar used for scaling the per point radius/footprint size.

    density_percentile_threshold (float): The density threshold as percentile.
        To remove 10% of the points with the lowest density set this to 10.

    point_radius_estimation_knn (int): The number of neighbors to use for 
        estimating the point radius/footprint size. This parameter is used if
        the radii array is empty.

    octree_max_depth (int): The maximum depth for the octree. This value should
        be in the range [6..21].

    contouring_value_threshold (float): The threshold for the unsigned distance
        for generating mesh vertices. For small values only vertices will be
        generated where the algorithm is certain about the surface. For large 
        values the algorithm will generate vertices even far away from the input
        data. The range of this value is [0..2] and the default value is 1.


    keep_n_connected_components (int): The number of connected components to 
        keep in the postprocessing.  Set this to \w n to keep only the \w n 
        largest connected components. By default all components will be kept.

    minimum_component_size (int): The minimum size of a component to be kept.
        The default is 3 which means even single triangles will be kept.

Returns:
    Returns the mesh as a dictionary with vertices and triangles.

)doc");

    m.def("remove_connected_components", &pyRemoveConnectedComponents,
          "vertices"_a, "triangles"_a, "keep_n_largest_components"_a,
          "minimum_component_size"_a = 3,
          R"doc(Removes small connected components from a triangle mesh.

Args:
    vertices (np.ndarray): Vertices with shape [num_vertices,3] and dtype
        np.float32.

    triangles (np.ndarray): Triangles with shape [num_triangles,3] and dtype
        np.int32.

    keep_n_largest_components (int): The number of largest connected components
        to keep. The returned mesh may still contain less components if the
        components have less than 'minimum_component_size' vertices.

    minimum_component_size (int): The minimum size of components that should be kept.

Returns:
    Returns a dictionary with vertices and triangles that contains the largest
    connected components.

)doc");

    m.def("create_octree", &pyCreateOctreeFromPoints, "points"_a, "radii"_a,
          "bb_min"_a, "bb_max"_a, "radius_scale"_a = 1, "grow_steps"_a = 0,
          "max_depth"_a = 21,
          R"doc(Creates an octree from a point cloud.

Args:
    points (np.ndarray): Points with shape [num_points, 3] and dtype np.float32.

    radii (np.ndarray): Per point radius describing the footprint size.
        The shape is [num_points] and the dtype is np.float32.

    bb_min (np.ndarray): The bounding box minimum corner with shape [3] and
        dtype np.float32.

    bb_max (np.ndarray): The bounding box maximum corner with shape [3] and
        dtype np.float32.

    radius_scale (float): Scalar that is used for scaling the radii.

    grow_steps (int): The number of steps used for growing the octree.
        This should be set to 0.

    max_depth (int): The maximum depth for the octree. This value should be in
        the range [6..21].

Returns:
    Returns a handle to an octree.

)doc");

    m.def("create_grids_from_octree", &pyCreateGridsFromOctree, "tree"_a,
          "num_levels"_a, "voxel_info_all_levels"_a = false,
          R"doc(Creates a grid hierarchy from an octree.

Args:
    tree (Octree): Octree as generated with create_octree_from_points.

    num_levels (int): The number of grid levels to generate.

    voxel_info_all_levels (bool): If False voxel information is only generated
        for the finest grid and coarse grids contain only neighbor connectivity
        information.

Returns:
    A list of grids starting with the finest grid. Each grid is a dict with:
    - voxel_keys: Contains the location codes for each voxel.
    - voxel_centers: The xyz position of each voxel center.
    - voxel_sizes: The size (edge length) of each voxel.
    - neighbors_index: Stores the indices of the face-adjacent voxel neighbors.
        The start and end position for each voxel is defined by the array
        'neighbors_row_splits'.
    - neighbors_kernel_index: The index for the kernel element to use in the 
        generalized sparse conv. Start and end for each voxel (row) is defined
        by the 'neighbors_row_splits' array.
    - neighbors_row_splits: Defines the start and end for each voxel (row) for
        the 'neighbors_index' and 'neighbors_kernel_index' arrays.
    - up_neighbors_index: Same as 'neighbors_index' but defines the connectivity
        from the current grid to the next finer grid.
    - up_neighbors_kernel_index: Same as 'neighbors_kernel_index' but defines
        the kernel element to use for the transition to the next finer grid.
    - up_neighbors_row_splits: Same as 'neighbors_row_splits' but for the
        'up_neighbors_index' and 'up_neighbors_kernel_index' arrays.

    Note that the keys starting with 'voxel_' are only defined for the finest
    grid by default. The information can be generated for all grids by setting
    'voxel_info_all_levels' to True.

    The keys starting with 'up_' do not exist for the finest grid.

)doc");

    m.def("create_dual_vertex_indices", &pyCreateDualVertexIndices, "tree"_a,
          R"doc(Creates an array with the vertex indices of the duals for the leaves of an octree.

Args:
    tree (Octree): Octree as generated with create_octree_from_points.

Returns:
    An array with shape [num_duals,8]. Each index corresponds to a voxel of the
    leaf nodes of the octree. A dual is described by up to 8 unique indices.

)doc");

    py::class_<pyKDTree, std::shared_ptr<pyKDTree>>(m, "KDTree")
            .def(py::init([](py::array_t<float, py::array::c_style> points) {
                     return std::make_shared<pyKDTree>(points);
                 }),
                 R"doc(KDTree with specialized functions for preprocessing.)doc")
            .def("compute_k_radius", &pyKDTree::ComputeKRadius, "k"_a,
                 R"doc(Computes the radius for each point that contains k points.)doc")
            .def("compute_inlier", &pyKDTree::ComputeInlier, "radii"_a,
                 "radius_fraction"_a = 0.5, "k"_a = 24,
                 "outlier_threshold"_a = 1,
                 R"doc(Simple inlier computation based on point radii.

For each point the radii of k neighbors is compared to a fraction of the current
point's radius to determine if the current point is an outlier.

Args:
    radii (np.ndarray): The radius for each point with shape [num_points].

    radius_fraction (float): If the radius of a neighbor is smaller than this
        fraction of the current point's radius then this neighbor counts
        towards the outlier threshold. This value should be  in the range (0..1).

    k (int): Number of neighbors to search.

    outlier_threshold (int): The number of neighbors that are outliers must be
        smaller than this value to count a point as inlier. This value should
        be in the range [1..k-1].

Returns:
    A bool vector indicating the inliers.

)doc")
            .def("compute_radius_neighbors", &pyKDTree::ComputeRadiusNeighbors,
                 "radii"_a,
                 R"doc(Counts how many neighbors are within a radius for each point.)doc");

}  // PYBIND11_MODULE