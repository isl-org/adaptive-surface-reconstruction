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
#include "contouring.h"

#include <array>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

#include "eigen.h"
#include "nested_vector.h"
#include "smallset.h"

ASR_NAMESPACE_BEGIN

void CreateTriangleMesh(std::vector<float>& vertices,
                        std::vector<int32_t>& triangles,
                        const std::vector<float>& values,
                        const std::vector<size_t> dual_indices,
                        const std::vector<float>& node_positions,
                        const float unsigned_threshold) {
    static_assert(sizeof(std::array<float, 3>) == 3 * sizeof(float));

    if (values.size() % 2) {
        throw std::runtime_error("values vector size is not a multiple of 2.");
    }

    if (dual_indices.size() % 8) {
        throw std::runtime_error(
                "dual_indices vector size is not a multiple of 8.");
    }

    if (node_positions.size() % 3) {
        throw std::runtime_error(
                "node_positions vector size is not a multiple of 3.");
    }

    triangles.clear();
    vertices.clear();

    //
    //         5      4
    //         +------+
    //      1  |   0  |
    //      +------+  |       y
    //      |  +---|--+       ^
    //      |  7   |  6       |
    //      +------+          +--> x
    //      3      2         /
    //                      V
    //                      z
    //
    const int cube_edges[12][2] = {{0, 1}, {1, 3}, {3, 2}, {2, 0},
                                   {4, 5}, {5, 7}, {7, 6}, {6, 4},
                                   {0, 4}, {1, 5}, {3, 7}, {2, 6}};
    const int cube_faces[6][4] = {
            {0, 1, 3, 2},  // front
            {4, 6, 7, 5},  // back
            {1, 5, 7, 3},  // left
            {2, 3, 7, 6},  // bottom
            {0, 2, 6, 4},  // right
            {0, 4, 5, 1}   // top
    };

    // the subset is {0,1}, {1,3}, {1,5}
    const int cube_edges_subset[3] = {0, 1, 9};

    auto ComplexSurfaceIntersectionTest =
            [&](const std::array<size_t, 8>& indices) {
                for (int edge_i = 0; edge_i < 12; ++edge_i) {
                    size_t edge_idx1 = indices[cube_edges[edge_i][0]];
                    size_t edge_idx2 = indices[cube_edges[edge_i][1]];
                    // unsigned values
                    float v1 = values[edge_idx1 * 2 + 1];
                    float v2 = values[edge_idx2 * 2 + 1];
                    if (v1 > unsigned_threshold && v2 > unsigned_threshold)
                        continue;
                    // signed values
                    v1 = values[edge_idx1 * 2 + 0];
                    v2 = values[edge_idx2 * 2 + 0];
                    if ((v1 < 0 && v2 > 0) || (v1 > 0 && v2 < 0)) return true;
                }
                return false;
            };

    auto EdgeIntersectionTest = [&](size_t edge_idx1, size_t edge_idx2) {
        if (edge_idx1 != edge_idx2) {
            // unsigned values
            float v1 = values[edge_idx1 * 2 + 1];
            float v2 = values[edge_idx2 * 2 + 1];
            if (v1 > unsigned_threshold && v2 > unsigned_threshold)
                return false;
            // signed values
            v1 = values[edge_idx1 * 2 + 0];
            v2 = values[edge_idx2 * 2 + 0];
            return (v1 < 0 && v2 > 0) || (v1 > 0 && v2 < 0);
        }
        return false;
    };

    auto ComputeVertexPosition = [&](const std::array<size_t, 8>& indices) {
        Vec3d vertex_pos(0, 0, 0);
        int count = 0;
        for (int edge_i = 0; edge_i < 12; ++edge_i) {
            size_t edge_idx1 = indices[cube_edges[edge_i][0]];
            size_t edge_idx2 = indices[cube_edges[edge_i][1]];
            // unsigned values
            double v1 = values[edge_idx1 * 2 + 1];
            double v2 = values[edge_idx2 * 2 + 1];
            if (v1 > unsigned_threshold && v2 > unsigned_threshold) continue;

            v1 = values[edge_idx1 * 2 + 0];
            v2 = values[edge_idx2 * 2 + 0];
            if ((v1 < 0 && v2 > 0) || (v1 > 0 && v2 < 0)) {
                Eigen::Map<const Vec3f> pos1(&node_positions[edge_idx1 * 3]);
                Eigen::Map<const Vec3f> pos2(&node_positions[edge_idx2 * 3]);

                double t = -v1 / (v2 - v1);
                if (!std::isfinite(t) || t < 0 || t > 1) t = 0.5;

                Vec3d intersection =
                        (1 - t) * pos1.cast<double>() + t * pos2.cast<double>();
                vertex_pos += intersection;
                ++count;
            }
        }
        vertex_pos /= count;
        return vertex_pos;
    };

    size_t num_out_vertices = 0;

    NestedVector<size_t> vertex_adjacent_duals;
    vertex_adjacent_duals.prefix_sum.resize(values.size() / 2, 0);
    // py::print("abc");

    for (size_t i = 0; i < dual_indices.size() / 8; ++i) {
        const std::array<size_t, 8>* dual =
                reinterpret_cast<const std::array<size_t, 8>*>(
                        &dual_indices[i * 8]);
        if (ComplexSurfaceIntersectionTest(*dual)) {
            ++num_out_vertices;
            std::unordered_set<size_t> tmp(dual->begin(), dual->end());

            for (size_t vidx : tmp) vertex_adjacent_duals.prefix_sum.at(vidx)++;
        }
    }

    // compute inclusive prefix sum
    std::partial_sum(vertex_adjacent_duals.prefix_sum.begin(),
                     vertex_adjacent_duals.prefix_sum.end(),
                     vertex_adjacent_duals.prefix_sum.begin());

    // py::print(num_out_vertices);
    // py::print(num_out_vertices);
    // py::print(num_out_vertices,"flush"_a=true);
    std::vector<size_t> intersecting_duals_indices(num_out_vertices);
    vertices.resize(num_out_vertices * 3);
    num_out_vertices = 0;  // recount

    // is a nested array which contains the indices to the
    // intersecting_duals_indices for all adjacent duals of this vertex
    vertex_adjacent_duals.data.resize(vertex_adjacent_duals.prefix_sum.back());

    {
        std::vector<size_t> vertex_dual_count(values.size() / 2, 0);
        for (size_t i = 0; i < dual_indices.size() / 8; ++i) {
            const std::array<size_t, 8>* dual =
                    reinterpret_cast<const std::array<size_t, 8>*>(
                            &dual_indices[i * 8]);
            if (ComplexSurfaceIntersectionTest(*dual)) {
                intersecting_duals_indices[num_out_vertices] = i;
                std::unordered_set<size_t> tmp(dual->begin(), dual->end());
                for (size_t vidx : tmp) {
                    size_t* ptr = vertex_adjacent_duals.getPtr(vidx);
                    ptr[vertex_dual_count[vidx]++] = num_out_vertices;
                }

                Vec3f v = ComputeVertexPosition(*dual).cast<float>();
                vertices[num_out_vertices * 3 + 0] = v.x();
                vertices[num_out_vertices * 3 + 1] = v.y();
                vertices[num_out_vertices * 3 + 2] = v.z();

                ++num_out_vertices;
            }
        }
    }

    auto GetDualsContainingEdge = [&](size_t evert1, size_t evert2) {
        std::unordered_set<size_t> set1;
        auto range1 = vertex_adjacent_duals.range(evert1);
        set1.insert(range1.first, range1.second);

        auto range2 = vertex_adjacent_duals.range(evert2);
        std::unordered_set<size_t> set2;
        for (auto it = range2.first; it != range2.second; ++it) {
            if (set1.count(*it)) set2.insert(*it);
        }
        return set2;
    };

    auto DualHasFace = [&](const std::array<size_t, 8>& dual,
                           const SmallSet<size_t, 4>& face) {
        for (int i = 0; i < 6; ++i) {
            SmallSet<size_t, 4> face_of_dual;
            for (int j = 0; j < 4; ++j) {
                face_of_dual.insert(dual[cube_faces[i][j]]);
            }
            if (face_of_dual == face) {
                return true;
            }
        }
        return false;
    };

    auto getDualFaceWithOrientedEdge = [&](const std::array<size_t, 8>& dual,
                                           const std::array<size_t, 2>& edge) {
        for (int face_i = 0; face_i < 6; ++face_i) {
            for (int j = 0; j < 4; ++j) {
                std::array<size_t, 2> face_edge{
                        dual[cube_faces[face_i][j]],
                        dual[cube_faces[face_i][(j + 1) % 4]]};
                if (face_edge == edge) {
                    SmallSet<size_t, 4> face;
                    for (int k = 0; k < 4; ++k)
                        face.insert(dual[cube_faces[face_i][k]]);
                    if (face.size() >= 3) return face;
                }
            }
        }
        return SmallSet<size_t, 4>();
    };

    auto sortDualsContainingEdge2 = [&](const std::unordered_set<size_t>&
                                                idx_set,
                                        std::array<size_t, 2> edge) {
        std::vector<size_t> idx_vec(idx_set.begin(), idx_set.end());
        // if(debug) py::print("=====", idx_vec);

        std::vector<size_t> sorted_idx_vec;
        sorted_idx_vec.reserve(idx_vec.size());
        sorted_idx_vec.push_back(idx_vec.back());
        idx_vec.pop_back();

        const int N = idx_set.size() * idx_set.size();

        size_t last_idx_vec_size = idx_vec.size();
        bool reverse_again = false;
        int reverse_count = 0;
        for (int i = 0; i < N && idx_vec.size(); ++i) {
            // find a dual complex which shares a face with the last dual
            // complex added to sorted_idx_vec
            const std::array<size_t, 8>* dual1 =
                    reinterpret_cast<const std::array<size_t, 8>*>(
                            &dual_indices[intersecting_duals_indices
                                                  [sorted_idx_vec.back()] *
                                          8]);
            auto face = getDualFaceWithOrientedEdge(*dual1, edge);

            last_idx_vec_size = idx_vec.size();
            for (auto it = idx_vec.begin(); it != idx_vec.end(); ++it) {
                const std::array<size_t, 8>* dual2 =
                        reinterpret_cast<const std::array<size_t, 8>*>(
                                &dual_indices[intersecting_duals_indices[*it] *
                                              8]);
                if (DualHasFace(*dual2, face)) {
                    sorted_idx_vec.push_back(*it);
                    idx_vec.erase(it);
                    break;
                }
            }

            // handle boundary edges by reversing to make sure that
            // we get all the duals in order.
            if (last_idx_vec_size == idx_vec.size()) {
                std::reverse(sorted_idx_vec.begin(), sorted_idx_vec.end());
                std::swap(edge[0], edge[1]);
                reverse_again = !reverse_again;
                ++reverse_count;
            }
        }
        // if(reverse_count) py::print(reverse_count);
        if (reverse_again)
            std::reverse(sorted_idx_vec.begin(), sorted_idx_vec.end());
        return sorted_idx_vec;
    };

    // check the subset of 3 edges and count the number of triangles we need to
    // create. The remaining edges will be checked by the adjacent dual
    // complexes.
    size_t num_triangles = 0;

    for (const size_t& dual_idx : intersecting_duals_indices) {
        const std::array<size_t, 8>* dual =
                reinterpret_cast<const std::array<size_t, 8>*>(
                        &dual_indices[dual_idx * 8]);
        for (int edge_i = 0; edge_i < 3; ++edge_i) {
            const size_t e_idx1 =
                    (*dual)[cube_edges[cube_edges_subset[edge_i]][0]];
            const size_t e_idx2 =
                    (*dual)[cube_edges[cube_edges_subset[edge_i]][1]];
            if (EdgeIntersectionTest(e_idx1, e_idx2)) {
                // create face
                auto duals_connected_by_this_edge =
                        GetDualsContainingEdge(e_idx1, e_idx2);

                int n = duals_connected_by_this_edge.size();
                if (n == 3) {
                    num_triangles += 1;
                } else if (n == 4) {
                    num_triangles += 2;
                } else if (n > 4) {
                    num_triangles += n;
                    num_out_vertices += 1;
                } else  // if( n < 3 )
                {
                    std::runtime_error(
                            "this should not happen: less than 3 vertices for "
                            "triangle");
                }
            }
        }
    }

    vertices.resize(num_out_vertices * 3);
    triangles.resize(num_triangles * 3);

    size_t triangle_idx = 0;
    size_t additional_vertex_idx = intersecting_duals_indices.size();

    for (const size_t& dual_idx : intersecting_duals_indices) {
        const std::array<size_t, 8>* dual =
                reinterpret_cast<const std::array<size_t, 8>*>(
                        &dual_indices[dual_idx * 8]);
        for (int edge_i = 0; edge_i < 3; ++edge_i) {
            size_t e_idx1 = (*dual)[cube_edges[cube_edges_subset[edge_i]][0]];
            size_t e_idx2 = (*dual)[cube_edges[cube_edges_subset[edge_i]][1]];
            if (EdgeIntersectionTest(e_idx1, e_idx2)) {
                // create face
                auto duals_connected_by_this_edge =
                        GetDualsContainingEdge(e_idx1, e_idx2);

                if (values[e_idx1 * 2] > values[e_idx2 * 2]) {
                    std::swap(e_idx1, e_idx2);
                }

                auto sorted_duals_connected_by_this_edge =
                        sortDualsContainingEdge2(duals_connected_by_this_edge,
                                                 {e_idx1, e_idx2});
                if (sorted_duals_connected_by_this_edge.size() !=
                    duals_connected_by_this_edge.size()) {
                    throw std::runtime_error(
                            "this should not happen: cannot sort duals");
                }

                int n = duals_connected_by_this_edge.size();

                std::vector<Vec3f> verts(n);
                for (int i = 0; i < n; ++i) {
                    if (sorted_duals_connected_by_this_edge[i] >=
                        vertices.size()) {
                        // py::print("this should not happen!!");
                    }
                    verts[i]
                            << vertices[sorted_duals_connected_by_this_edge[i] *
                                                3 +
                                        0],
                            vertices[sorted_duals_connected_by_this_edge[i] *
                                             3 +
                                     1],
                            vertices[sorted_duals_connected_by_this_edge[i] *
                                             3 +
                                     2];
                }

                if (n == 3) {
                    triangles[triangle_idx * 3 + 0] =
                            sorted_duals_connected_by_this_edge[0];
                    triangles[triangle_idx * 3 + 1] =
                            sorted_duals_connected_by_this_edge[1];
                    triangles[triangle_idx * 3 + 2] =
                            sorted_duals_connected_by_this_edge[2];
                    triangle_idx += 1;
                } else if (n == 4) {
                    if ((verts[0] - verts[2]).squaredNorm() >
                        (verts[1] - verts[3]).squaredNorm()) {
                        triangles[(triangle_idx + 0) * 3 + 0] =
                                sorted_duals_connected_by_this_edge[0];
                        triangles[(triangle_idx + 0) * 3 + 1] =
                                sorted_duals_connected_by_this_edge[1];
                        triangles[(triangle_idx + 0) * 3 + 2] =
                                sorted_duals_connected_by_this_edge[3];

                        triangles[(triangle_idx + 1) * 3 + 0] =
                                sorted_duals_connected_by_this_edge[1];
                        triangles[(triangle_idx + 1) * 3 + 1] =
                                sorted_duals_connected_by_this_edge[2];
                        triangles[(triangle_idx + 1) * 3 + 2] =
                                sorted_duals_connected_by_this_edge[3];
                    } else {
                        triangles[(triangle_idx + 0) * 3 + 0] =
                                sorted_duals_connected_by_this_edge[0];
                        triangles[(triangle_idx + 0) * 3 + 1] =
                                sorted_duals_connected_by_this_edge[1];
                        triangles[(triangle_idx + 0) * 3 + 2] =
                                sorted_duals_connected_by_this_edge[2];

                        triangles[(triangle_idx + 1) * 3 + 0] =
                                sorted_duals_connected_by_this_edge[0];
                        triangles[(triangle_idx + 1) * 3 + 1] =
                                sorted_duals_connected_by_this_edge[2];
                        triangles[(triangle_idx + 1) * 3 + 2] =
                                sorted_duals_connected_by_this_edge[3];
                    }
                    triangle_idx += 2;
                } else if (n > 4) {
                    Vec3f center_vertex(0, 0, 0);
                    for (int i = 0; i < n; ++i) {
                        center_vertex += verts[i];
                    }
                    center_vertex /= sorted_duals_connected_by_this_edge.size();
                    size_t center_vertex_idx = additional_vertex_idx++;
                    vertices[center_vertex_idx * 3 + 0] = center_vertex.x();
                    vertices[center_vertex_idx * 3 + 1] = center_vertex.y();
                    vertices[center_vertex_idx * 3 + 2] = center_vertex.z();

                    for (int i = 0; i < n; ++i) {
                        std::array<size_t, 3> t = {
                                sorted_duals_connected_by_this_edge[i],
                                sorted_duals_connected_by_this_edge[(i + 1) %
                                                                    n],
                                center_vertex_idx};
                        triangles[(triangle_idx + i) * 3 + 0] = t[0];
                        triangles[(triangle_idx + i) * 3 + 1] = t[1];
                        triangles[(triangle_idx + i) * 3 + 2] = t[2];
                    }
                    triangle_idx += n;
                } else  // if( n < 3 )
                {
                    std::runtime_error(
                            "this should not happen2: less than 3 vertices for "
                            "triangle");
                }
            }
        }
    }
}

ASR_NAMESPACE_END
