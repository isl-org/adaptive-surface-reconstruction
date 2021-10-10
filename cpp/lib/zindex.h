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
#include <cmath>
#include <cstdint>
#include <vector>

#include "asr_config.h"

ASR_NAMESPACE_BEGIN

/*!
 *  Computes the morton number for three 21-bit integers
 *
 *  \param x  Integer that uses up to 21 bit
 *  \param y  Integer that uses up to 21 bit
 *  \param z  Integer that uses up to 21 bit
 *
 *  \return   The morton number as 64-bit int with 63 bits used.
 */
inline uint64_t Morton3d(uint64_t x, uint64_t y, uint64_t z) {
    x = (x | (x << 32)) & 0x001F00000000FFFF;
    x = (x | (x << 16)) & 0x00FF0000FF0000FF;
    x = (x | (x << 8)) & 0xF00F00F00F00F00F;
    x = (x | (x << 4)) & 0x30C30C30C30C30C3;
    x = (x | (x << 2)) & 0x9249249249249249;

    y = (y | (y << 32)) & 0x001F00000000FFFF;
    y = (y | (y << 16)) & 0x00FF0000FF0000FF;
    y = (y | (y << 8)) & 0xF00F00F00F00F00F;
    y = (y | (y << 4)) & 0x30C30C30C30C30C3;
    y = (y | (y << 2)) & 0x9249249249249249;

    z = (z | (z << 32)) & 0x001F00000000FFFF;
    z = (z | (z << 16)) & 0x00FF0000FF0000FF;
    z = (z | (z << 8)) & 0xF00F00F00F00F00F;
    z = (z | (z << 4)) & 0x30C30C30C30C30C3;
    z = (z | (z << 2)) & 0x9249249249249249;

    return x | (y << 1) | (z << 2);
}

/*!
 *  Computes the morton number for three 10-bit integers
 *
 *  \param x  Integer that uses up to 10 bit
 *  \param y  Integer that uses up to 10 bit
 *  \param z  Integer that uses up to 10 bit
 *
 *  \return   The morton number as 32-bit int with 30 bits used.
 */
inline uint32_t Morton3d(uint32_t x, uint32_t y, uint32_t z) {
    x = (x | (x << 16)) & 0x030000FF;
    x = (x | (x << 8)) & 0x0300F00F;
    x = (x | (x << 4)) & 0x030C30C3;
    x = (x | (x << 2)) & 0x09249249;

    y = (y | (y << 16)) & 0x030000FF;
    y = (y | (y << 8)) & 0x0300F00F;
    y = (y | (y << 4)) & 0x030C30C3;
    y = (y | (y << 2)) & 0x09249249;

    z = (z | (z << 16)) & 0x030000FF;
    z = (z | (z << 8)) & 0x0300F00F;
    z = (z | (z << 4)) & 0x030C30C3;
    z = (z | (z << 2)) & 0x09249249;

    return x | (y << 1) | (z << 2);
}

/*!
 *  Computes the non-interleaved inputs from the given morton number
 *
 *  \param x      Output parameter. stores as 21-bit integer
 *  \param y      Output parameter. stores as 21-bit integer
 *  \param z      Output parameter. stores as 21-bit integer
 *  \param input  Input morton number with 63 bits. MSB must be 0.
 */
inline void InverseMorton3d(uint64_t& x,
                            uint64_t& y,
                            uint64_t& z,
                            uint64_t input) {
    x = input & 0x1249249249249249;
    y = (input >> 1) & 0x1249249249249249;
    z = (input >> 2) & 0x1249249249249249;

    x = ((x >> 2) | x) & 0x30C30C30C30C30C3;
    x = ((x >> 4) | x) & 0xF00F00F00F00F00F;
    x = ((x >> 8) | x) & 0x00FF0000FF0000FF;
    x = ((x >> 16) | x) & 0x001F00000000FFFF;
    x = ((x >> 32) | x) & 0x00000000001FFFFF;

    y = ((y >> 2) | y) & 0x30C30C30C30C30C3;
    y = ((y >> 4) | y) & 0xF00F00F00F00F00F;
    y = ((y >> 8) | y) & 0x00FF0000FF0000FF;
    y = ((y >> 16) | y) & 0x001F00000000FFFF;
    y = ((y >> 32) | y) & 0x00000000001FFFFF;

    z = ((z >> 2) | z) & 0x30C30C30C30C30C3;
    z = ((z >> 4) | z) & 0xF00F00F00F00F00F;
    z = ((z >> 8) | z) & 0x00FF0000FF0000FF;
    z = ((z >> 16) | z) & 0x001F00000000FFFF;
    z = ((z >> 32) | z) & 0x00000000001FFFFF;
}

/*!
 *  Computes the non-interleaved inputs from the given morton number
 *
 *  \param x      Output parameter. stores as 10-bit integer
 *  \param y      Output parameter. stores as 10-bit integer
 *  \param z      Output parameter. stores as 10-bit integer
 *  \param input  Input morton number with 30 bits. The two most significant
 *                bits must be 0.
 */
inline void InverseMorton3d(uint32_t& x,
                            uint32_t& y,
                            uint32_t& z,
                            uint32_t input) {
    x = input & 0x09249249;
    y = (input >> 1) & 0x09249249;
    z = (input >> 2) & 0x09249249;

    x = ((x >> 2) | x) & 0x030C30C3;
    x = ((x >> 4) | x) & 0x0300F00F;
    x = ((x >> 8) | x) & 0x030000FF;
    x = ((x >> 16) | x) & 0x000003FF;

    y = ((y >> 2) | y) & 0x030C30C3;
    y = ((y >> 4) | y) & 0x0300F00F;
    y = ((y >> 8) | y) & 0x030000FF;
    y = ((y >> 16) | y) & 0x000003FF;

    z = ((z >> 2) | z) & 0x030C30C3;
    z = ((z >> 4) | z) & 0x0300F00F;
    z = ((z >> 8) | z) & 0x030000FF;
    z = ((z >> 16) | z) & 0x000003FF;
}

/*!
 *  Adds two morton numbers
 *  \param a  A dilated integer as obtained from motion_3d
 *  \param b  A dilated integer as obtained from motion_3d
 *  \return   The dilated sum of the morton numbers
 */
inline uint64_t MortonAdd3d(uint64_t a, uint64_t b) {
    uint64_t c;
    c = ((a | ~0x9249249249249249) + (b & 0x9249249249249249)) &
        0x9249249249249249;
    c |= ((a | ~(0x9249249249249249 << 1)) + (b & (0x9249249249249249 << 1))) &
         (0x9249249249249249 << 1);
    c |= ((a | ~(0x9249249249249249 << 2)) + (b & (0x9249249249249249 << 2))) &
         (0x9249249249249249 << 2);
    return c;
}

/*!
 *  Adds two morton numbers
 *  \param a  A dilated integer as obtained from motion_3d
 *  \param b  A dilated integer as obtained from motion_3d
 *  \return   The dilated sum of the morton numbers
 */
inline uint32_t MortonAdd3d(uint32_t a, uint32_t b) {
    uint32_t c;
    c = ((a | ~0x49249249) + (b & 0x49249249)) & 0x49249249;
    c |= ((a | ~(0x49249249 << 1)) + (b & (0x49249249 << 1))) &
         (0x49249249 << 1);
    c |= ((a | ~(0x09249249 << 2)) + (b & (0x09249249 << 2))) &
         (0x09249249 << 2);
    return c;
}

/*!
 *  Subtracts two morton numbers
 *  \param a  A dilated integer as obtained from motion_3d
 *  \param b  A dilated integer as obtained from motion_3d
 *  \return   The subtraction of the morton numbers
 */
inline uint64_t MortonSub3d(uint64_t a, uint64_t b) {
    uint64_t c;
    c = ((a & 0x9249249249249249) - (b & 0x9249249249249249)) &
        0x9249249249249249;
    c |= ((a & (0x9249249249249249 << 1)) - (b & (0x9249249249249249 << 1))) &
         (0x9249249249249249 << 1);
    c |= ((a & (0x9249249249249249 << 2)) - (b & (0x9249249249249249 << 2))) &
         (0x9249249249249249 << 2);
    return c;
}

/*!
 *  Subtracts two morton numbers
 *  \param a  A dilated integer as obtained from motion_3d
 *  \param b  A dilated integer as obtained from motion_3d
 *  \return   The subtraction of the morton numbers
 */
inline uint32_t MortonSub3d(uint32_t a, uint32_t b) {
    uint32_t c;
    c = ((a & 0x49249249) - (b & 0x49249249)) & 0x49249249;
    c |= ((a & (0x49249249 << 1)) - (b & (0x49249249 << 1))) &
         (0x49249249 << 1);
    c |= ((a & (0x09249249 << 2)) - (b & (0x09249249 << 2))) &
         (0x09249249 << 2);
    return c;
}

ASR_NAMESPACE_END