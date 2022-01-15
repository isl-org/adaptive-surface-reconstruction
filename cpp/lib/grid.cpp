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
#include "grid.h"

#include <numeric>

ASR_NAMESPACE_BEGIN

void ConvertKeysToIndices(
        std::vector<size_t>& dualverts_index,
        const std::vector<std::array<Octree::KeyType, 8>>& duals,
        const std::vector<Octree::KeyType>& sorted_leaves) {
    dualverts_index.resize(duals.size() * 8);
    for (size_t i = 0; i < duals.size(); ++i) {
        const auto& d = duals[i];
        for (int vertex_i = 0; vertex_i < 8; ++vertex_i) {
            auto iter = std::lower_bound(sorted_leaves.begin(),
                                         sorted_leaves.end(), d[vertex_i]);
            if (iter != sorted_leaves.end()) {
                size_t idx = std::distance(sorted_leaves.begin(), iter);
                dualverts_index[i * 8 + vertex_i] = idx;
            } else {
                throw std::runtime_error(
                        "cannot compute index. cannot find leaf node");
            }
        }
    }
}

void CreateLeafNeighborInformation(std::vector<int32_t>& neighbors_index,
                                   std::vector<uint8_t>& neighbors_kernel_index,
                                   std::vector<int64_t>& neighbors_row_splits,
                                   const std::vector<Octree::KeyType>& keys) {
    typedef Octree::KeyType KeyType;
    typedef Octree::Coord Coord;

    neighbors_index.clear();
    neighbors_kernel_index.clear();
    neighbors_row_splits.resize(keys.size() + 1);

    std::vector<Vec3i> neighbor_offsets({Vec3i(-1, 0, 0), Vec3i(1, 0, 0),
                                         Vec3i(0, -1, 0), Vec3i(0, 1, 0),
                                         Vec3i(0, 0, -1), Vec3i(0, 0, 1)});

    std::vector<std::array<int, 8>> parent_kernel_index_offset(
            {{-1, 0, -1, 1, -1, 2, -1, 3},
             {0, -1, 1, -1, 2, -1, 3, -1},
             {-1, -1, 0, 1, -1, -1, 2, 3},
             {0, 1, -1, -1, 2, 3, -1, -1},
             {-1, -1, -1, -1, 0, 1, 2, 3},
             {0, 1, 2, 3, -1, -1, -1, -1}});

    std::vector<Vec3i> child_neighbor_offsets(
            {Vec3i(-1, 0, 0), Vec3i(-1, 1, 0),
             Vec3i(-1, 0, 1), Vec3i(-1, 1, 1),

             Vec3i(2, 0, 0),  Vec3i(2, 1, 0),
             Vec3i(2, 0, 1),  Vec3i(2, 1, 1),

             Vec3i(0, -1, 0), Vec3i(1, -1, 0),
             Vec3i(0, -1, 1), Vec3i(1, -1, 1),

             Vec3i(0, 2, 0),  Vec3i(1, 2, 0),
             Vec3i(0, 2, 1),  Vec3i(1, 2, 1),

             Vec3i(0, 0, -1), Vec3i(1, 0, -1),
             Vec3i(0, 1, -1), Vec3i(1, 1, -1),

             Vec3i(0, 0, 2),  Vec3i(1, 0, 2),
             Vec3i(0, 1, 2),  Vec3i(1, 1, 2)});

    auto find_node = [&](KeyType k) {
        // search is always successful
        auto iter = std::lower_bound(keys.begin(), keys.end(), k);
        if (keys.end() != iter && *iter == k) {
            return int64_t(std::distance(keys.begin(), iter));
        }
        return int64_t(-1);
    };

    for (size_t i = 0; i < keys.size(); ++i) {
        const KeyType key = keys[i];
        const int level = Octree::ComputeLevel(key);
        const Coord coord = Octree::ComputeCoord(key);

        int16_t kernel_idx = 0;
        int num_neighbors = 0;

        // add the voxel itself
        ++num_neighbors;
        neighbors_index.push_back(i);
        neighbors_kernel_index.push_back(kernel_idx);
        ++kernel_idx;

        // search same level face adjacent nodes
        for (size_t j = 0; j < neighbor_offsets.size(); ++j) {
            Coord neighbor_coord(coord);
            neighbor_coord.x += neighbor_offsets[j].x();
            neighbor_coord.y += neighbor_offsets[j].y();
            neighbor_coord.z += neighbor_offsets[j].z();
            KeyType neighbor_key = Octree::ComputeKey(neighbor_coord);
            if (Octree::INVALID_KEY() != neighbor_key) {
                int64_t idx = find_node(neighbor_key);
                if (idx >= 0) {
                    ++num_neighbors;
                    neighbors_index.push_back(idx);
                    neighbors_kernel_index.push_back(kernel_idx);
                }
            }
            ++kernel_idx;
        }

        // search face adjacent nodes on the child level
        if (level < Octree::MAX_LEVEL()) {
            for (size_t j = 0; j < child_neighbor_offsets.size(); ++j) {
                Coord neighbor_coord = Octree::ComputeCoord(key << 3);
                neighbor_coord.x += child_neighbor_offsets[j].x();
                neighbor_coord.y += child_neighbor_offsets[j].y();
                neighbor_coord.z += child_neighbor_offsets[j].z();
                KeyType neighbor_key = Octree::ComputeKey(neighbor_coord);
                if (Octree::INVALID_KEY() != neighbor_key) {
                    int64_t idx = find_node(neighbor_key);
                    if (idx >= 0) {
                        ++num_neighbors;
                        neighbors_index.push_back(idx);
                        neighbors_kernel_index.push_back(kernel_idx);
                    }
                }
                ++kernel_idx;
            }
        } else {
            kernel_idx += child_neighbor_offsets.size();
        }

        // search face adjacent nodes on the parent level
        if (level > 0) {
            for (size_t j = 0; j < neighbor_offsets.size(); ++j) {
                Coord neighbor_coord(coord);
                neighbor_coord.x += neighbor_offsets[j].x();
                neighbor_coord.y += neighbor_offsets[j].y();
                neighbor_coord.z += neighbor_offsets[j].z();
                KeyType neighbor_key = Octree::ComputeKey(neighbor_coord);
                if (Octree::INVALID_KEY() != neighbor_key) {
                    int configuration = neighbor_key & 7;
                    int64_t idx = find_node(neighbor_key >> 3);
                    if (idx >= 0) {
                        ++num_neighbors;
                        neighbors_index.push_back(idx);
                        int16_t k_idx =
                                kernel_idx +
                                parent_kernel_index_offset[j][configuration];
                        neighbors_kernel_index.push_back(k_idx);
                    }
                }
                kernel_idx += 4;
            }
        }

        neighbors_row_splits[i + 1] =
                int64_t(num_neighbors) + neighbors_row_splits[i];
    }
}

void CombineSiblings(std::vector<Octree::KeyType>& out_keys,
                     std::vector<int32_t>& neighbors_index,
                     std::vector<uint8_t>& neighbors_kernel_index,
                     std::vector<int64_t>& neighbors_row_splits,
                     const std::vector<Octree::KeyType>& keys) {
    typedef Octree::KeyType KeyType;
    out_keys.clear();
    for (size_t i = 0; i < keys.size(); ++i) {
        KeyType key = keys[i];
        if ((key & 7) == 0) {
            int num_siblings = 0;
            for (size_t j = i + 1; j < keys.size(); ++j) {
                if ((keys[j] & ~KeyType(7)) == key)
                    ++num_siblings;
                else
                    break;
            }
            if (7 == num_siblings) {
                out_keys.push_back(key >> 3);
                i += 7;  // skip the other 7 siblings
            } else {
                out_keys.push_back(key);
            }
        } else {
            out_keys.push_back(key);
        }
    }
    std::sort(out_keys.begin(), out_keys.end());

    neighbors_row_splits.resize(keys.size() + 1);
    std::iota(neighbors_row_splits.begin(), neighbors_row_splits.end(), 0);
    neighbors_index.resize(keys.size());
    neighbors_kernel_index.resize(keys.size());

    auto find_out_key_idx = [&](KeyType k) {
        // search is always successful
        auto iter = std::lower_bound(out_keys.begin(), out_keys.end(), k);
        return int64_t(std::distance(out_keys.begin(), iter));
    };

    for (size_t i = 0; i < keys.size(); ++i) {
        KeyType key = keys[i];
        if ((key & 7) == 0) {
            int num_siblings = 0;
            for (size_t j = i + 1; j < keys.size(); ++j) {
                if ((keys[j] & ~KeyType(7)) == key)
                    ++num_siblings;
                else
                    break;
            }
            if (7 == num_siblings) {
                int64_t out_key_idx = find_out_key_idx(key >> 3);
                for (int j = 0; j < 8; ++j) {
                    neighbors_index[i + j] = out_key_idx;
                    neighbors_kernel_index[i + j] = j;
                }
                i += 7;  // skip the other 7 siblings
            } else {
                neighbors_index[i] = find_out_key_idx(key);
                neighbors_kernel_index[i] = 8;
            }
        } else {
            neighbors_index[i] = find_out_key_idx(key);
            neighbors_kernel_index[i] = 8;
        }
    }
}

std::vector<std::shared_ptr<ASRGrid>> CreateGridsFromOctree(
        const Octree& tree,
        const int num_levels,
        const bool voxel_info_all_levels) {
    typedef Octree::KeyType KeyType;

    auto InitGridVoxelInfo = [&](const std::vector<KeyType>& keys,
                                 ASRGrid& grid) {
        grid.voxel_keys = keys;
        grid.voxel_centers.resize(3 * keys.size());
        grid.voxel_sizes.resize(keys.size());

        size_t i = 0;
        for (const auto& key : keys) {
            Vec3f p = tree.ComputeVoxelCenter(key);
            grid.voxel_centers[i * 3 + 0] = p.x();
            grid.voxel_centers[i * 3 + 1] = p.y();
            grid.voxel_centers[i * 3 + 2] = p.z();

            int lev = tree.ComputeLevel(key);
            grid.voxel_sizes[i] = tree.VoxelSize(lev);
            ++i;
        }
    };

    // create finest grid
    std::vector<std::shared_ptr<ASRGrid>> result;
    {
        auto grid = std::make_shared<ASRGrid>();
        InitGridVoxelInfo(tree.leaves, *grid);

        CreateLeafNeighborInformation(grid->neighbors_index,
                                      grid->neighbors_kernel_index,
                                      grid->neighbors_row_splits, tree.leaves);

        result.push_back(grid);
    }

    // create coarser grids with additional connectivity info between the coarse
    // and the adjacent finer grid.
    std::vector<KeyType> nodes;
    std::vector<KeyType> nodes2;
    for (int i = 1; i < num_levels; ++i) {
        auto& prev_grid = result.back();
        auto grid = std::make_shared<ASRGrid>();

        if (i == 1) {
            CombineSiblings(nodes, prev_grid->up_neighbors_index,
                            prev_grid->up_neighbors_kernel_index,
                            prev_grid->up_neighbors_row_splits, tree.leaves);
        } else {
            CombineSiblings(nodes, prev_grid->up_neighbors_index,
                            prev_grid->up_neighbors_kernel_index,
                            prev_grid->up_neighbors_row_splits, nodes2);
        }

        if (voxel_info_all_levels) {
            InitGridVoxelInfo(nodes, *grid);
        }

        CreateLeafNeighborInformation(grid->neighbors_index,
                                      grid->neighbors_kernel_index,
                                      grid->neighbors_row_splits, nodes);

        nodes2.swap(nodes);
        result.push_back(grid);
    }

    return result;
}

size_t CountDualCells(const Octree& tree,
                      const std::vector<Octree::KeyType>& leaves) {
    typedef Octree::KeyType KeyType;

    size_t cell_count = 0;

    for (const KeyType& node_key : leaves) {
        const auto vertex_keys = tree.ComputeVertexKeys(node_key);

        for (int i = 0; i < 8; ++i) {
            KeyType vertex_key = vertex_keys[i];
            if (vertex_key == tree.INVALID_KEY()) continue;

            bool skip_vertex = false;

            const auto adjacent_node_keys =
                    tree.ComputeAdjacentNodeKeys(vertex_key);

            for (int j = 0; j < 8; ++j) {
                if (i == j) continue;
                const KeyType adjnode_key = adjacent_node_keys[j];

                Octree::Attributes attr;
                bool node_found = tree.hashmap.find(adjnode_key, attr.ui);

                // if there is no node then node_key is deeper
                // -> check next adjacent node
                if (!node_found) continue;

                // if we find a node that is not a leaf then one of the
                // childs of that node will create the dual complex for
                // this vertex -> go to the next vertex
                if (!attr.leaf) {
                    skip_vertex = true;
                    break;
                }

                // the adjacent node is a leaf with the same level.
                // if the index of the adjacent node is smaller than the current
                // node then the adjacent node will create the dual complex
                if (adjnode_key < node_key) {
                    skip_vertex = true;
                    break;
                }
            }

            if (!skip_vertex) {
                ++cell_count;
            }
        }
    }

    return cell_count;
}

void CreateDualCells(std::vector<std::array<Octree::KeyType, 8>>&
                             duals,  // dual defined by the 8 keys that
                                     // correspond to voxel centers
                     std::vector<Octree::KeyType>&
                             dual_primal_vertices,  // the primal vertices
                                                    // within the dual cells
                     const Octree& tree,
                     const std::vector<Octree::KeyType>& leaves) {
    typedef Octree::KeyType KeyType;
    // py::print("root", tree.hashmap.contains(1));

    const size_t cell_count = CountDualCells(tree, leaves);
    duals.resize(cell_count);
    dual_primal_vertices.resize(cell_count);

    size_t out_idx = 0;

    for (const KeyType& node_key : leaves) {
        // const int voxel_level = tree.computeLevel(center_voxel_key);
        const auto vertex_keys = tree.ComputeVertexKeys(node_key);

        for (int i = 0; i < 8; ++i) {
            KeyType vertex_key = vertex_keys[i];
            if (vertex_key == tree.INVALID_KEY()) continue;

            bool skip_vertex = false;

            auto adjacent_node_keys = tree.ComputeAdjacentNodeKeys(vertex_key);

            for (int j = 0; j < 8; ++j) {
                if (i == j) continue;
                const KeyType adjnode_key = adjacent_node_keys[j];

                Octree::Attributes attr;
                bool node_found = tree.hashmap.find(adjnode_key, attr.ui);

                // if there is no node then node_key is deeper
                // -> check next adjacent node
                if (!node_found) continue;

                // if we find a node that is not a leaf then one of the
                // childs of that node will create the dual complex for
                // this vertex -> go to the next vertex
                if (!attr.leaf) {
                    skip_vertex = true;
                    break;
                }

                // the adjacent node is a leaf with the same level.
                // if the index of the adjacent node is smaller than the current
                // node then the adjacent node will create the dual complex
                if (adjnode_key < node_key) {
                    skip_vertex = true;
                    break;
                }
            }

            if (!skip_vertex) {
                for (KeyType& key : adjacent_node_keys) {
                    Octree::Attributes attr{0};
                    while (tree.INVALID_KEY() != key &&
                           !tree.hashmap.find(key, attr.ui)) {
                        key >>= 3;
                    }
                    if (tree.INVALID_KEY() == key) {
                        throw std::runtime_error(
                                "invalid key after searching for node");
                    } else if (!attr.leaf) {
                        throw std::runtime_error("found node is not a leaf");
                    }
                }
                duals[out_idx] = adjacent_node_keys;
                dual_primal_vertices[out_idx] = vertex_key;
                ++out_idx;
            }
        }
    }
}

void CreateDualVertexIndices(std::vector<size_t>& dual_vertex_indices,
                             const Octree& tree) {
    typedef Octree::KeyType KeyType;

    std::vector<std::array<KeyType, 8>> duals;
    std::vector<KeyType> dual_primal_vertices_keys;
    CreateDualCells(duals, dual_primal_vertices_keys, tree, tree.leaves);

    ConvertKeysToIndices(dual_vertex_indices, duals, tree.leaves);
}

ASR_NAMESPACE_END