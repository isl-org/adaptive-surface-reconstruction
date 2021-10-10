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

#include "asr_config.h"
#include "eigen.h"
#include "octreebase.h"

ASR_NAMESPACE_BEGIN

class Octree : public OctreeBase<uint64_t, unsigned int> {
public:
    typedef OctreeBase<uint64_t, unsigned int> BaseClass;
    typedef BaseClass::Coord Coord;
    typedef BaseClass::KeyType KeyType;
    typedef BaseClass::ValueType ValueType;

    union Attributes {
        unsigned int ui;
        struct {
            unsigned char leaf;
        };
    };

    inline Octree(const Vec3f& bb_min, const Vec3f& bb_max, float scale_bb);

    Octree() {}

    int ComputeLevelFromScale(float scale) const {
        for (int level = 0; level <= MAX_LEVEL(); ++level) {
            if (voxel_size[level] < scale) return std::max(0, level - 1);
        }
        return MAX_LEVEL();
    }

    Coord ComputeCoord(const Vec3f& pos, int level = MAX_LEVEL()) const {
        Coord coord;
        level = std::min(MAX_LEVEL(), level);
        coord.x = std::floor(pos.x() * inv_voxel_size[MAX_LEVEL()]);
        coord.y = std::floor(pos.y() * inv_voxel_size[MAX_LEVEL()]);
        coord.z = std::floor(pos.z() * inv_voxel_size[MAX_LEVEL()]);
        coord.lev = level;
        coord.x += offset.x();
        coord.y += offset.y();
        coord.z += offset.z();
        coord.x >>= MAX_LEVEL() - coord.lev;
        coord.y >>= MAX_LEVEL() - coord.lev;
        coord.z >>= MAX_LEVEL() - coord.lev;
        return coord;
    }

    KeyType ComputeKey(const Vec3f& pos, int level = MAX_LEVEL()) const {
        Coord coord = ComputeCoord(pos, level);
        return BaseClass::ComputeKey(coord);
    }

    static KeyType ComputeKey(const Coord& coord) {
        return BaseClass::ComputeKey(coord);
    }

    static Coord ComputeCoord(KeyType key) {
        return BaseClass::ComputeCoord(key);
    }

    Vec3f ComputeVoxelCenter(const Coord& coord) const {
        Vec3f center;
        int s = MAX_LEVEL() - coord.lev;
        Vec3i tmp;
        tmp.x() = coord.x << s;
        tmp.y() = coord.y << s;
        tmp.z() = coord.z << s;

        center.x() = ((tmp.x() - offset.x()) + 0.5f * std::pow(2, s)) *
                     voxel_size[MAX_LEVEL()];
        center.y() = ((tmp.y() - offset.y()) + 0.5f * std::pow(2, s)) *
                     voxel_size[MAX_LEVEL()];
        center.z() = ((tmp.z() - offset.z()) + 0.5f * std::pow(2, s)) *
                     voxel_size[MAX_LEVEL()];
        return center;
    }

    Vec3f ComputeVoxelCenter(const KeyType& key) const {
        return ComputeVoxelCenter(BaseClass::ComputeCoord(key));
    }

    Vec3f ComputeVertexPosition(const Coord& coord) const {
        Vec3f center;
        int s = MAX_LEVEL() - coord.lev;
        Vec3i tmp;
        tmp.x() = coord.x << s;
        tmp.y() = coord.y << s;
        tmp.z() = coord.z << s;

        center.x() = (tmp.x() - offset.x()) * voxel_size[MAX_LEVEL()];
        center.y() = (tmp.y() - offset.y()) * voxel_size[MAX_LEVEL()];
        center.z() = (tmp.z() - offset.z()) * voxel_size[MAX_LEVEL()];
        return center;
    }

    Vec3f ComputeVertexPosition(const KeyType& vertex_key) const {
        return ComputeVertexPosition(BaseClass::ComputeCoord(vertex_key));
    }

    float VoxelSize(const int& level = MAX_LEVEL()) const {
        return voxel_size[level];
    }

    float InvVoxelSize(const int& level = MAX_LEVEL()) const {
        return inv_voxel_size[level];
    }

    void Grow(const int iterations);

    void CreateAncestorsAndSiblings();

    void BalanceFaces();

    void InitAttributesAndLeaves();

    Vec3i offset;

    Vec3f original_bb_min;
    Vec3f original_bb_max;

    std::vector<KeyType> leaves;

private:
    float voxel_size[MAX_LEVEL() + 1];
    float inv_voxel_size[MAX_LEVEL() + 1];
};

std::shared_ptr<Octree> CreateOctreeFromPoints(const float* const points,
                                               const size_t num_points,
                                               const float* const radii,
                                               const Vec3f& bb_min,
                                               const Vec3f& bb_max,
                                               const float radius_scale,
                                               const int grow_steps,
                                               const int max_depth);

ASR_NAMESPACE_END