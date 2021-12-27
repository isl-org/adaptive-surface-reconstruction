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
#include "postprocess.h"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <unordered_set>

#include "nested_vector.h"

ASR_NAMESPACE_BEGIN

void ComputeVertexFaces(NestedVector<int64_t>& vertex_faces,
                        const std::vector<float>& vertices,
                        std::vector<int32_t>& triangles) {
    vertex_faces.clear();
    if (0 == vertices.size()) return;
    const size_t n_vertices = vertices.size() / 3;
    const size_t n_triangles = triangles.size() / 3;
    // count the adjacent faces for each vertex
    std::vector<size_t>& vertex_face_count = vertex_faces.prefix_sum;
    vertex_face_count.resize(n_vertices, 0);
    size_t max_vertex_face_count = 0;
    for (size_t i = 0; i < n_triangles; ++i) {
        std::array<int32_t, 3> tri({
                triangles[i * 3 + 0],
                triangles[i * 3 + 1],
                triangles[i * 3 + 2],
        });
        for (auto v_idx : tri) {
            if (++vertex_face_count[v_idx] > max_vertex_face_count)
                ++max_vertex_face_count;
        }
    }
    // build the prefix sum
    std::partial_sum(vertex_face_count.begin(), vertex_face_count.end(),
                     vertex_face_count.begin());
    vertex_faces.data.resize(vertex_faces.prefix_sum.back(), -1);

    // insert the face indices
    for (size_t face_idx = 0; face_idx < n_triangles; ++face_idx) {
        std::array<int32_t, 3> tri({
                triangles[face_idx * 3 + 0],
                triangles[face_idx * 3 + 1],
                triangles[face_idx * 3 + 2],
        });
        for (auto tri_vidx : tri) {
            auto vertex_face_range = vertex_faces.range(tri_vidx);
            bool insert_success = false;
            for (auto it2 = vertex_face_range.first;
                 it2 != vertex_face_range.second; ++it2) {
                if (-1 == *it2) {
                    insert_success = true;
                    (*it2) = face_idx;
                    break;
                }
            }
            if (!insert_success) {
                throw std::runtime_error(
                        "this should not happen: too many faces inserted for "
                        "vertex");
            }
        }
    }
}

void ConnectedComponents(std::vector<int64_t>& component,
                         std::vector<int64_t>& component_sizes,
                         const std::vector<float>& vertices,
                         std::vector<int32_t>& triangles) {
    if (vertices.size() % 3) {
        throw std::runtime_error(
                "vertices vector size is not a multiple of 3.");
    }
    if (triangles.size() % 3) {
        throw std::runtime_error(
                "triangles vector size is not a multiple of 3.");
    }
    const size_t num = vertices.size() / 3;

    component.clear();
    component.resize(num, -1);
    component_sizes.clear();

    NestedVector<int64_t> vertex_faces;
    ComputeVertexFaces(vertex_faces, vertices, triangles);

    std::vector<ssize_t> todo_list;
    int64_t current_component = 0;
    for (size_t i = 0; i < num; ++i) {
        if (-1 != component[i]) continue;

        todo_list.push_back(i);
        while (todo_list.size()) {
            int v_idx = todo_list.back();
            todo_list.pop_back();

            // todo_list may contain duplicates
            if (-1 != component[v_idx]) continue;

            component[v_idx] = current_component;

            // add neighbours
            std::unordered_set<int64_t> neighbours;
            auto faces_range = vertex_faces.range(v_idx);
            for (auto it = faces_range.first; it != faces_range.second; ++it) {
                std::array<int32_t, 3> tri({
                        triangles[*it * 3 + 0],
                        triangles[*it * 3 + 1],
                        triangles[*it * 3 + 2],
                });
                for (auto it2 : tri) neighbours.insert(it2);
            }

            for (int neighbour_idx : neighbours)
                if (-1 == component[neighbour_idx])
                    todo_list.push_back(neighbour_idx);
        }
        ++current_component;
    }

    component_sizes.resize(current_component, 0);

    for (size_t i = 0; i < num; ++i) ++component_sizes[component[i]];
}

void RemoveConnectedComponents(std::vector<float>& vertices,
                               std::vector<int32_t>& triangles,
                               int64_t keep_n_largest_components,
                               int64_t minimum_component_size) {
    std::vector<int64_t> component;
    std::vector<int64_t> component_sizes;

    ConnectedComponents(component, component_sizes, vertices, triangles);

    std::vector<std::pair<int64_t, int64_t>> component_sizes_idx;
    for (int64_t idx = 0; idx < int64_t(component_sizes.size()); ++idx) {
        component_sizes_idx.push_back(
                std::make_pair(component_sizes[idx], idx));
    }
    std::sort(component_sizes_idx.begin(), component_sizes_idx.end(),
              std::greater<>());

    std::unordered_set<int64_t> components_to_keep;
    for (int64_t i = 0; i < std::min(int64_t(component_sizes.size()),
                                     keep_n_largest_components);
         ++i) {
        if (component_sizes_idx[i].first >= minimum_component_size) {
            components_to_keep.insert(component_sizes_idx[i].second);
        }
    }

    std::vector<int8_t> mask(component.size(), 0);
    for (size_t i = 0; i < component.size(); ++i) {
        if (components_to_keep.count(component[i])) {
            mask[i] = 1;
        }
    }

    FilterTriangleMeshVertices(vertices, triangles, mask);
}

void FilterTriangleMeshVertices(std::vector<float>& vertices,
                                std::vector<int32_t>& triangles,
                                const std::vector<int8_t>& mask) {
    std::vector<size_t> prefix_sum(mask.size() + 1, 0);
    for (size_t i = 0; i < mask.size(); ++i) {
        prefix_sum[i + 1] = prefix_sum[i] + (mask[i] ? 1 : 0);
        if (mask[i]) {
            vertices[prefix_sum[i] * 3 + 0] = vertices[i * 3 + 0];
            vertices[prefix_sum[i] * 3 + 1] = vertices[i * 3 + 1];
            vertices[prefix_sum[i] * 3 + 2] = vertices[i * 3 + 2];
        }
    }
    vertices.resize(prefix_sum.back() * 3);

    std::vector<int32_t> tmp_triangles;
    for (size_t i = 0; i < triangles.size() / 3; ++i) {
        if (mask[triangles[i * 3 + 0]] && mask[triangles[i * 3 + 1]] &&
            mask[triangles[i * 3 + 2]]) {
            tmp_triangles.push_back(prefix_sum[triangles[i * 3 + 0]]);
            tmp_triangles.push_back(prefix_sum[triangles[i * 3 + 1]]);
            tmp_triangles.push_back(prefix_sum[triangles[i * 3 + 2]]);
        }
    }
    triangles.swap(tmp_triangles);
}

ASR_NAMESPACE_END
