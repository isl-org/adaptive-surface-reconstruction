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
#include "octree.h"

ASR_NAMESPACE_BEGIN

Octree::Octree(const Vec3f& bb_min, const Vec3f& bb_max, float scale_bb)
    : original_bb_min(bb_min), original_bb_max(bb_max) {
    Vec3f bb_center = 0.5f * (bb_max + bb_min);

    float cube_edge_length = bb_max.x() - bb_min.x();
    cube_edge_length = std::max(cube_edge_length, bb_max.y() - bb_min.y());
    cube_edge_length = std::max(cube_edge_length, bb_max.z() - bb_min.z());
    cube_edge_length *= scale_bb;

    // recompute bb
    Vec3f new_bb_min = bb_center.array() - 0.5f * cube_edge_length;

    voxel_size[0] = cube_edge_length;
    inv_voxel_size[0] = 1 / cube_edge_length;
    for (int i = 1; i <= MAX_LEVEL(); ++i) {
        double tmp = cube_edge_length * (1.0 / std::pow(2, i));
        voxel_size[i] = tmp;
        inv_voxel_size[i] = 1.0 / tmp;
    }
    offset.x() = -std::floor(new_bb_min.x() * inv_voxel_size[MAX_LEVEL()]);
    offset.y() = -std::floor(new_bb_min.y() * inv_voxel_size[MAX_LEVEL()]);
    offset.z() = -std::floor(new_bb_min.z() * inv_voxel_size[MAX_LEVEL()]);
}

void Octree::Grow(const int iterations) {
    if (0 == iterations) return;

    Vec3i neighbor_offsets[8];
    neighbor_offsets[0] = Vec3i(0, 0, 0);
    neighbor_offsets[1] = Vec3i(1, 0, 0);
    neighbor_offsets[2] = Vec3i(0, 1, 0);
    neighbor_offsets[3] = Vec3i(1, 1, 0);
    neighbor_offsets[4] = Vec3i(0, 0, 1);
    neighbor_offsets[5] = Vec3i(1, 0, 1);
    neighbor_offsets[6] = Vec3i(0, 1, 1);
    neighbor_offsets[7] = Vec3i(1, 1, 1);

    std::vector<KeyType> keys_to_process;
    {
        auto lt = hashmap.lock_table();
        keys_to_process.reserve(lt.size());
        for (const auto& it : lt) keys_to_process.push_back(it.first);
    }
    std::vector<KeyType> keys_to_process2;

    for (int iteration = 0; iteration < iterations; ++iteration) {
        for (size_t i = 0; i < keys_to_process.size(); ++i) {
            const KeyType current_key = keys_to_process[i];
            if (1 == current_key)  // skip root node
                continue;

            const KeyType parent_key = current_key >> 3;
            const Coord parent_coord = ComputeCoord(parent_key);

            const int configuration = 7 - (current_key & 7);
            Vec3i offset = neighbor_offsets[configuration];
            for (int j = 0; j < 8; ++j) {
                if (configuration == j)
                    continue;  // the current_key node has a parent -> skip
                Vec3i tmp = neighbor_offsets[j] - offset;
                Coord coord;
                coord.x = parent_coord.x + tmp.x();
                coord.y = parent_coord.y + tmp.y();
                coord.z = parent_coord.z + tmp.z();
                coord.lev = parent_coord.lev;

                KeyType key = ComputeKey(coord);
                if (INVALID_KEY() == key) continue;

                unsigned int dummy = 0;
                if (!hashmap.find(key, dummy) && !HasChild(key)) {
                    hashmap.insert_or_assign(key, 0);
                    keys_to_process2.push_back(key);
                }
            }

            // create the siblings
            KeyType first_sibling_key = current_key & ~KeyType(7);
            for (int j = 0; j < 8; ++j) {
                KeyType sibling_key = first_sibling_key + j;
                if (current_key == sibling_key) continue;
                if (hashmap.insert(sibling_key, 0))
                    keys_to_process2.push_back(sibling_key);
            }
        }
        keys_to_process.clear();
        keys_to_process2.swap(keys_to_process);
    }
}

void Octree::CreateAncestorsAndSiblings() {
    std::vector<KeyType> keys_to_process;
    {
        auto lt = hashmap.lock_table();
        keys_to_process.reserve(lt.size());
        for (const auto& it : lt) keys_to_process.push_back(it.first);
    }
    std::vector<KeyType> keys_to_process2;

    while (keys_to_process.size()) {
        // create siblings
        for (size_t i = 0; i < keys_to_process.size(); ++i) {
            KeyType key = keys_to_process[i];

            if (1 == key) continue;  // root node does not have siblings

            KeyType first_sibling_key = key & ~KeyType(7);
            for (int j = 0; j < 8; ++j) {
                KeyType sibling_key = first_sibling_key + j;
                if (key == sibling_key) continue;
                if (hashmap.insert(sibling_key, 0) &&
                    sibling_key == first_sibling_key)
                    keys_to_process2.push_back(sibling_key);
            }
            if (key == first_sibling_key) keys_to_process2.push_back(key);
        }
        keys_to_process.clear();
        keys_to_process2.swap(keys_to_process);

        for (size_t i = 0; i < keys_to_process.size(); ++i) {
            KeyType key = keys_to_process[i];
            key >>= 3;
            if (INVALID_KEY() != key) {
                hashmap.insert(key, 0);
                keys_to_process2.push_back(key);
            }
        }
        keys_to_process.clear();
        keys_to_process2.swap(keys_to_process);
    }
}

void Octree::BalanceFaces() {
    std::vector<Vec3i> neighbor_offsets({Vec3i(-1, 0, 0), Vec3i(0, -1, 0),
                                         Vec3i(0, 0, -1), Vec3i(1, 0, 0),
                                         Vec3i(0, 1, 0), Vec3i(0, 0, 1)});

    std::vector<KeyType> keys_to_process;
    {
        auto lt = hashmap.lock_table();
        keys_to_process.reserve(lt.size());
        for (const auto& it : lt) {
            // there are no mixed nodes; process only the first sibling
            if (!(it.first & 7)) keys_to_process.push_back(it.first);
        }
    }
    std::vector<KeyType> keys_to_process2;

    while (keys_to_process.size()) {
        for (size_t i = 0; i < keys_to_process.size(); ++i) {
            const KeyType current_key = keys_to_process[i];
            if (1 == current_key)  // skip root node
                continue;

            if (HasFirstChild(current_key))  // not a leaf
                continue;
            const Coord current_parent_coord = ComputeCoord(current_key >> 3);

            for (const Vec3i& offset : neighbor_offsets) {
                Coord coord;
                coord.x = current_parent_coord.x + offset.x();
                coord.y = current_parent_coord.y + offset.y();
                coord.z = current_parent_coord.z + offset.z();
                coord.lev = current_parent_coord.lev;

                KeyType key = ComputeKey(coord);
                if (INVALID_KEY() == key) continue;

                unsigned int dummy;
                while (!hashmap.find(key, dummy)) {
                    // create all siblings
                    KeyType first_sibling_key = key & ~KeyType(7);
                    if (hashmap.insert(first_sibling_key, 0)) {
                        keys_to_process2.push_back(first_sibling_key);
                    }
                    for (int j = 1; j < 8; ++j) {
                        KeyType sibling_key = first_sibling_key + j;
                        hashmap.insert(sibling_key, 0);
                    }
                    key >>= 3;
                }
            }
        }
        keys_to_process.clear();
        keys_to_process2.swap(keys_to_process);
    }
}

void Octree::InitAttributesAndLeaves() {
    leaves.clear();

    std::vector<KeyType> keys;
    {
        auto lt = hashmap.lock_table();
        keys.reserve(lt.size());
        for (const auto& it : lt) keys.push_back(it.first);
    }

    for (const KeyType& key : keys) {
        Attributes attr{0};

        if (!HasFirstChild(key)) {
            attr.leaf = 1;
            leaves.push_back(key);
        }
        hashmap.update(key, attr.ui);
    }
    std::sort(leaves.begin(), leaves.end());
}

std::shared_ptr<Octree> CreateOctreeFromPoints(const float* const points,
                                               const size_t num_points,
                                               const float* const radii,
                                               const Vec3f& bb_min,
                                               const Vec3f& bb_max,
                                               const float radius_scale,
                                               const int grow_steps,
                                               const int max_depth) {
    typedef Octree::KeyType KeyType;

    std::shared_ptr<Octree> tree(new Octree(bb_min, bb_max, 1));

    int min_level = Octree::MAX_LEVEL();
    int max_level = 0;
    for (size_t i = 0; i < num_points; ++i) {
        Vec3f p(points[i * 3 + 0], points[i * 3 + 1], points[i * 3 + 2]);
        float radius = radii[i];

        if ((p.array() < bb_min.array()).any() ||
            (p.array() > bb_max.array()).any()) {
            continue;
        }

        int level = tree->ComputeLevelFromScale(radius_scale * radius);
        level = std::min(max_depth, level);
        min_level = std::min(min_level, level);
        max_level = std::max(max_level, level);
        KeyType key = tree->ComputeKey(p, level);
        tree->hashmap.insert(key, 0);
    }
    // py::print("a", tree->checkAllKeys());

    // py::print("mixed nodes", tree_has_mixed_nodes(*tree));
    tree->Grow(grow_steps);
    // py::print("b", tree->checkAllKeys());
    // py::print("mixed nodes", tree_has_mixed_nodes(*tree));
    // removeAncestors(*tree, 0);
    // VisualizeTreeLeaves("tree",0, *tree);
    tree->CreateAncestorsAndSiblings();
    // py::print("mixed nodes", tree_has_mixed_nodes(*tree));
    // py::print("c", tree->checkAllKeys());

    // VisualizeTreeLeaves("tree",1, *tree);
    tree->BalanceFaces();
    // VisualizeTreeLeaves("tree",2, *tree);

    tree->InitAttributesAndLeaves();
    // py::print("d", tree->checkAllKeys());

    return tree;
}

ASR_NAMESPACE_END