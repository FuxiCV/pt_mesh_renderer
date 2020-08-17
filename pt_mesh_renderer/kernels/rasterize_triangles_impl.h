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

// Modifications: this files integrates forward and backward codes togather 
// under PyTorch c++ style
// Copyright 2020 Netease Fuxi AI LAB
// SPDX-License-Identifier: Apache-2.0

#ifndef MESH_RENDERER_KERNELS_RASTERIZE_TRIANGLES_IMPL_H_
#define MESH_RENDERER_KERNELS_RASTERIZE_TRIANGLES_IMPL_H_

namespace pytorch_mesh_renderer
{
    // Computes the triangle id, barycentric coordinates, and z-buffer at each pixel
    // in the image.
    //
    // vertices: A flattened 2D array with 4*vertex_count elements.
    //     Each contiguous triplet is the XYZW location of the vertex with that
    //     triplet's id. The coordinates are assumed to be OpenGL-style clip-space
    //     (i.e., post-projection, pre-divide), where X points right, Y points up,
    //     Z points away.
    // triangles: A flattened 2D array with 3*triangle_count elements.
    //     Each contiguous triplet is the three vertex ids indexing into vertices
    //     describing one triangle with clockwise winding.
    // triangle_count: The number of triangles stored in the array triangles.
    // triangle_ids: A flattened 2D array with image_height*image_width elements.
    //     At return, each pixel contains a triangle id in the range
    //     [0, triangle_count). The id value is also 0 if there is no triangle
    //     at the pixel. The barycentric_coordinates must be checked to
    //     distinguish the two cases.
    // barycentric_coordinates: A flattened 3D array with
    //     image_height*image_width*3 elements. At return, contains the triplet of
    //     barycentric coordinates at each pixel in the same vertex ordering as
    //     triangles. If no triangle is present, all coordinates are 0.
    // z_buffer: A flattened 2D array with image_height*image_width elements. At
    //     return, contains the normalized device Z coordinates of the rendered
    //     triangles.
    void RasterizeTrianglesForward(
        const float* vertices, const int64_t* triangles,
        int64_t triangle_count, int32_t image_width,
        int32_t image_height, int64_t* triangle_ids,
        float* barycentric_coordinates, float* z_buffer);

    // Threshold for a barycentric coordinate triplet's sum, below which the
    // coordinates at a pixel are deemed degenerate. Most such degenerate triplets
    // in an image will be exactly zero, as this is how pixels outside the mesh
    // are rendered.
    constexpr float kDegenerateBarycentricCoordinatesCutoff = 0.9f;

    // If the area of a triangle is very small in screen space, the corner vertices
    // are approaching colinearity, and we should drop the gradient to avoid
    // numerical instability (in particular, blowup, as the forward pass computation
    // already only has 8 bits of precision).
    constexpr float kMinimumTriangleArea = 1e-13f;

    // The derivative of RasterizeTrianglesForward function
    void RasterizeTrianglesBackward(
        const float* vertices, const int64_t* triangles,
        const float* barycentric_coordinates,
        const int64_t* triangle_ids,
        const float* df_dbarycentric_coordinates,
        int32_t image_width, int32_t image_height,
        float* df_dvertices);

} // namespace pytorch_mesh_renderer
#endif // MESH_RENDERER_OPS_KERNELS_RASTERIZE_TRIANGLES_IMPL_H_
