// Copyright 2017 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Modifications: PyTorch c++ interface implementation
// Copyright 2020 Netease Fuxi AI LAB
// SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>
#include <algorithm>
#include <vector>
#include "rasterize_triangles_impl.h"

using namespace pytorch_mesh_renderer;

#define CHECK_CPU(x) AT_ASSERTM(!(x.type().is_cuda()), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CPU(x);      \
    CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> forward_rasterize_triangles(
    at::Tensor vertices,  // FloatTensor
    at::Tensor triangles, // LongTensor
    int32_t image_width,
    int32_t image_height)
{
    CHECK_INPUT(vertices);
    CHECK_INPUT(triangles);

    int64_t triangle_count = triangles.sizes()[0];

    torch::Tensor barycentric = torch::zeros({ image_height, image_width, 3 }, torch::kFloat32);
    torch::Tensor triangle_ids = torch::zeros({ image_height, image_width }, torch::kInt64);
    torch::Tensor z_buffer = torch::ones({ image_height, image_width }, torch::kFloat32);

    RasterizeTrianglesForward(vertices.data<float>(), triangles.data<int64_t>(), triangle_count,
        image_width, image_height, triangle_ids.data<int64_t>(), barycentric.data<float>(), z_buffer.data<float>());
    return { barycentric, triangle_ids, z_buffer };
}

std::vector<torch::Tensor> backward_rasterize_triangles(
    at::Tensor vertices,                    // FloatTensor
    at::Tensor triangles,                   // LongTensor
    at::Tensor barycentric_coordinates,     // FloatTensor
    at::Tensor triangle_ids,                // LongTensor
    at::Tensor df_dbarycentric_coordinates, // FloatTensor
    int32_t image_width,
    int32_t image_height)
{
    CHECK_INPUT(vertices);
    CHECK_INPUT(triangles);
    CHECK_INPUT(barycentric_coordinates);
    CHECK_INPUT(triangle_ids);
    CHECK_INPUT(df_dbarycentric_coordinates);
    torch::Tensor df_dvertices = torch::zeros({ vertices.sizes()[0], 4 }, torch::kFloat32);

    RasterizeTrianglesBackward(vertices.data<float>(), triangles.data<int64_t>(),
        barycentric_coordinates.data<float>(), triangle_ids.data<int64_t>(),
        df_dbarycentric_coordinates.data<float>(),
        image_width, image_height, df_dvertices.data<float>());
    return { df_dvertices };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward_rasterize_triangles", &forward_rasterize_triangles, "FORWARD_RASTERIZE_TRIANGLES (CPU)");
    m.def("backward_rasterize_triangles", &backward_rasterize_triangles, "BACKWARD_RASTERIZE_TRIANGLES (CPU)");
}
