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
#include <open3d/t/io/TriangleMeshIO.h>
#include <open3d/utility/Console.h>

#include <iostream>

#include "asr.hpp"
#include "eigen.h"
#include "plyreader.h"

void ReadPoints(std::vector<float>& points,
                std::vector<float>& normals,
                std::vector<float>& radii,
                const std::string& path) {
    std::cout << "reading points\n";

    PlyReader ply(path);
    ply.readHeader();
    const PlyReader::Element* vertex = ply.getElement("vertex");

    ssize_t num_points = 0;
    if (vertex) {
        num_points = vertex->count;
    }

    if (!num_points) {
        return;
    }

    size_t point_idx = 0;

    auto ElementCallback = [&](size_t, size_t global_idx) {
        point_idx = global_idx;
        if (!(point_idx % 1000000)) {
            std::cout << point_idx << " / " << num_points << std::endl;
        }
    };

    auto PropertyCallback = [&](std::vector<double>& values, int id) {
        double value = values.front();
        switch (id) {
            case 0:
                points[point_idx * 3 + 0] = value;
                break;
            case 1:
                points[point_idx * 3 + 1] = value;
                break;
            case 2:
                points[point_idx * 3 + 2] = value;
                break;
            case 3:
                normals[point_idx * 3 + 0] = value;
                break;
            case 4:
                normals[point_idx * 3 + 1] = value;
                break;
            case 5:
                normals[point_idx * 3 + 2] = value;
                break;
            case 6:
                radii[point_idx] = value;
                break;
            default:
                break;
        }
    };

    bool required = true;
    required = required && ply.setupCallback(ElementCallback, "vertex");
    required =
            required && ply.setupCallback(PropertyCallback, "vertex", "x", 0);
    required =
            required && ply.setupCallback(PropertyCallback, "vertex", "y", 1);
    required =
            required && ply.setupCallback(PropertyCallback, "vertex", "z", 2);
    required =
            required && ply.setupCallback(PropertyCallback, "vertex", "nx", 3);
    required =
            required && ply.setupCallback(PropertyCallback, "vertex", "ny", 4);
    required =
            required && ply.setupCallback(PropertyCallback, "vertex", "nz", 5);

    if (!required) {
        return;
    }

    points.resize(3 * num_points);
    normals.resize(3 * num_points);

    bool has_radii = ply.setupCallback(PropertyCallback, "vertex", "value", 6);
    has_radii = has_radii ||
                ply.setupCallback(PropertyCallback, "vertex", "radius", 6);

    if (has_radii) {
        radii.resize(num_points);
    }

    ply.readElements("vertex");
    std::cout << point_idx << " / " << num_points << std::endl;
}

void PrintHelp() {
    std::cout <<
            R"(usage: asrtool --in point_cloud.ply --out mesh.ply

Arguments:
    in      Input point cloud with normal information in PLY format.
    out     Output mesh in PLY format.

Options:
    --version  Prints the version information
    --third-party-notices  Prints third-party software notices
)";
};

int main(int argc, char** argv) {
    if (open3d::utility::ProgramOptionExists(argc, argv, "--version")) {
        std::cout << "asrtool version " << asr::GetVersionStr() << "\n";
        return 0;
    }

    if (open3d::utility::ProgramOptionExists(argc, argv,
                                             "--third-party-notices")) {
        std::cout << asr::GetThirdPartyNotices() << "\n";
        return 0;
    }

    if (!open3d::utility::ProgramOptionExists(argc, argv, "--in") ||
        !open3d::utility::ProgramOptionExists(argc, argv, "--out")) {
        PrintHelp();
        return 1;
    }

    auto input_path =
            open3d::utility::GetProgramOptionAsString(argc, argv, "--in");
    auto output_path =
            open3d::utility::GetProgramOptionAsString(argc, argv, "--out");

    std::vector<float> points, normals, radii;
    ReadPoints(points, normals, radii, input_path);

    auto print_callback = [](const std::string& s) { std::cout << s; };
    asr::SetPrintCallbackFunction(print_callback,
                                  {asr::INFO, asr::WARN, asr::ERROR});

    auto params = asr::InitReconstructSurfaceParams();

    auto result = asr::ReconstructSurface(points, normals, radii, params);

    {
        open3d::core::Tensor vertices(result->vertices,
                                      {int64_t(result->vertices.size() / 3), 3},
                                      open3d::core::Dtype::Float32);
        open3d::core::Tensor triangles(
                result->triangles, {int64_t(result->triangles.size() / 3), 3},
                open3d::core::Dtype::Int32);

        open3d::t::geometry::TriangleMesh tmesh(vertices, triangles);
        open3d::t::io::WriteTriangleMesh(output_path, tmesh);
    }

    return 0;
}