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

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

ASR_NAMESPACE_BEGIN

// convenience typedefs

typedef Eigen::Array4f Arr4f;

typedef Eigen::Vector2i Vec2i;
typedef Eigen::Vector2f Vec2f;
typedef Eigen::Vector2d Vec2d;

typedef Eigen::Matrix<unsigned char, 3, 1> Vec3uchar;
typedef Eigen::Vector3i Vec3i;
typedef Eigen::Vector3f Vec3f;
typedef Eigen::Vector3d Vec3d;

typedef Eigen::Matrix<unsigned char, 4, 1> Vec4uchar;
typedef Eigen::Vector4i Vec4i;
typedef Eigen::Vector4f Vec4f;
typedef Eigen::Vector4d Vec4d;

typedef Eigen::Matrix3f Mat3f;
typedef Eigen::Matrix3d Mat3d;

typedef Eigen::Matrix4f Mat4f;
typedef Eigen::Matrix4d Mat4d;

typedef Eigen::MatrixXf MatXf;
typedef Eigen::MatrixXd MatXd;

template <class T>
struct EigenSTLContainers {
    typedef std::vector<T, Eigen::aligned_allocator<T>> vector;
};

typedef EigenSTLContainers<Arr4f>::vector std_vector_Arr4f;

typedef EigenSTLContainers<Vec4uchar>::vector std_vector_Vec4uchar;

typedef EigenSTLContainers<Vec4f>::vector std_vector_Vec4f;
typedef EigenSTLContainers<Vec4i>::vector std_vector_Vec4i;
typedef EigenSTLContainers<Vec2f>::vector std_vector_Vec2f;
typedef EigenSTLContainers<Vec2i>::vector std_vector_Vec2i;

ASR_NAMESPACE_END