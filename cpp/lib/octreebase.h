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

#include "libcuckoo/cuckoohash_map.hh"
#include "zindex.h"

ASR_NAMESPACE_BEGIN

template <class KEY = uint64_t, class VALUE = unsigned int>
class OctreeBase {
    static_assert(std::is_same<KEY, uint64_t>::value ||
                          std::is_same<KEY, uint32_t>::value,
                  "unsupported key type for OctreeBase");

public:
    OctreeBase() {}
    virtual ~OctreeBase() {}

    struct Coord {
        int x, y, z, lev;
    };
    typedef KEY KeyType;
    typedef VALUE ValueType;
    typedef libcuckoo::cuckoohash_map<KeyType, ValueType> HashMap;

    constexpr static int MIN_LEVEL() { return 0; }
    constexpr static int MAX_LEVEL() { return (sizeof(KeyType) * 8) / 3; }
    constexpr static KeyType INVALID_KEY() { return 0; }
    constexpr static KeyType INTERLEAVE_MASK() {
        if (std::is_same<KeyType, uint32_t>::value) {
            return 0x49249249;
        } else if (std::is_same<KeyType, uint64_t>::value) {
            return 0x9249249249249249;
        }
        return 0;
    }

    static int ComputeLevel(const KeyType& key) {
        return (MAX_LEVEL() * 3 - CountLeadingZeros(key) +
                int(std::is_same<KeyType, uint32_t>::value)) /
               3;
    }

    static KeyType ComputeKey(const Coord& coord) {
        KeyType key;
        if (!IsValidCoord(coord)) return INVALID_KEY();
        key = Morton3d(KeyType(coord.x), KeyType(coord.y), KeyType(coord.z)) |
              (KeyType(1) << 3 * coord.lev);
        return key;
    }

    static Coord ComputeCoord(KeyType key) {
        Coord coord;
        coord.lev = ComputeLevel(key);

        KeyType x, y, z;
        InverseMorton3d(x, y, z, key & ~(KeyType(1) << coord.lev * 3));
        coord.x = x;
        coord.y = y;
        coord.z = z;
        return coord;
    }

    static KeyType MinLevelKey(int level) { return KeyType(1) << (3 * level); }

    static KeyType MaxLevelKey(int level) {
        return ~KeyType(0) >> (3 * (MAX_LEVEL() - level) +
                               int(std::is_same<KeyType, uint32_t>::value));
    }

    static std::array<KeyType, 8> ComputeVertexKeys(KeyType key) {
        std::array<KeyType, 8> result = {0, 0, 0, 0,
                                         0, 0, 0, 0};  // invalid keys
        int lev = ComputeLevel(key);
        KeyType min_lev_key = MinLevelKey(lev);

        for (KeyType i = 0; i < 8; ++i) {
            KeyType vk;
            vk = MortonAdd3d(key, i);

            // remove the leading bit
            KeyType vk_ = vk - min_lev_key;
            if (vk >= (min_lev_key << 1) || !(vk_ & INTERLEAVE_MASK()) ||
                !(vk_ & INTERLEAVE_MASK() << 1) ||
                !(vk_ & INTERLEAVE_MASK() << 2)) {
                continue;
            }
            result[i] = vk;
        }
        return result;
    }

    static std::array<KeyType, 8> ComputeAdjacentNodeKeys(KeyType vertex_key) {
        std::array<KeyType, 8> result = {0, 0, 0, 0,
                                         0, 0, 0, 0};  // invalid keys

        for (KeyType i = 0; i < 8; ++i) {
            KeyType nk;
            nk = MortonSub3d(vertex_key, i);
            result[i] = nk;
        }
        return result;
    }

    static bool IsValidCoord(const Coord& coord) {
        if (coord.lev > MAX_LEVEL() || coord.x < 0 ||
            coord.x >= (int32_t(1) << coord.lev) || coord.y < 0 ||
            coord.y >= (int32_t(1) << coord.lev) || coord.z < 0 ||
            coord.z >= (int32_t(1) << coord.lev)) {
            return false;
        }
        return true;
    }

    static bool IsValidKey(const KeyType& key) {
        if (key == INVALID_KEY()) return false;
        int lz = CountLeadingZeros(key) -
                 int(std::is_same<KeyType, uint32_t>::value);
        if (lz % 3) return false;
        return true;
    }

    bool HasChild(const KeyType& key,
                  typename HashMap::locked_table& lt) const {
        // a key with level MAX_LEVEL has either no leading zeros (64 bit keys)
        // or a single leading zero (32 bit keys)
        if (CountLeadingZeros(key) <= 1) return false;
        for (int k = 0; k < 8; ++k) {
            KeyType child_key = (key << 3) + k;
            if (lt.count(child_key)) return true;
        }
        return false;
    }

    bool IsMixedNode(const KeyType& key) const {
        // a key with level MAX_LEVEL has either no leading zeros (64 bit keys)
        // or a single leading zero (32 bit keys)
        if (CountLeadingZeros(key) <= 1) return false;
        int child_count = 0;
        for (int k = 0; k < 8; ++k) {
            KeyType child_key = (key << 3) + k;
            if (hashmap.contains(child_key)) ++child_count;
            if (child_count != 0 && child_count < k + 1) return true;
        }
        return false;
    }

    bool HasChild(const KeyType& key) const {
        // a key with level MAX_LEVEL has either no leading zeros (64 bit keys)
        // or a single leading zero (32 bit keys)
        if (CountLeadingZeros(key) <= 1) return false;
        for (int k = 0; k < 8; ++k) {
            KeyType child_key = (key << 3) + k;
            if (hashmap.contains(child_key)) return true;
        }
        return false;
    }

    bool HasFirstChild(const KeyType& key) const {
        // a key with level MAX_LEVEL has either no leading zeros (64 bit keys)
        // or a single leading zero (32 bit keys)
        if (CountLeadingZeros(key) <= 1) return false;
        KeyType child_key = (key << 3);
        if (hashmap.contains(child_key)) return true;
        return false;
    }

    bool CheckAllKeys() {
        auto lt = hashmap.lock_table();
        for (const auto& it : lt) {
            if (!isValidKey(it.first)) return false;
        }
        return true;
    }

    HashMap hashmap;

private:
    static int CountLeadingZeros(const uint32_t& value) {
        return __builtin_clz(value);
    }
    static int CountLeadingZeros(const uint64_t& value) {
        return __builtin_clzl(value);
    }
};

ASR_NAMESPACE_END