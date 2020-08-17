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

// Modifications: CUDA implementation of CPU verison
// Copyright 2020 Netease Fuxi AI LAB
// SPDX-License-Identifier: Apache-2.0

#include "rasterize_triangles_cuda_impl.h"


namespace pytorch_mesh_renderer
{
    // Takes the minimum of a, b, and c, rounds down, and converts to an integer
    // in the range [low, high].
    template <typename scalar_t>
    __device__ __forceinline__ int32_t ClampedIntegerMin(scalar_t a, scalar_t b, scalar_t c, int32_t low, int32_t high)
    {
        return (int32_t)fmin(fmax((floor(fmin(fmin(a, b), c))), (scalar_t)low), (scalar_t)high);
    }

    // Takes the maximum of a, b, and c, rounds up, and converts to an integer
    // in the range [low, high].
    template <typename scalar_t>
    __device__ __forceinline__ int32_t ClampedIntegerMax(scalar_t a, scalar_t b, scalar_t c, int32_t low, int32_t high)
    {
        return (int32_t)fmin(fmax((ceil(fmax(fmax(a, b), c))), (scalar_t)low), (scalar_t)high);
    }

    // Computes a 3x3 matrix inverse without dividing by the determinant.
    // Instead, makes an unnormalized matrix inverse with the correct sign
    // by flipping the sign of the matrix if the determinant is negative.
    // By leaving out determinant division, the rows of M^-1 only depend on two out
    // of three of the columns of M; i.e., the first row of M^-1 only depends on the
    // second and third columns of M, the second only depends on the first and
    // third, etc. This means we can compute edge functions for two neighboring
    // triangles independently and produce exactly the same numerical result up to
    // the sign. This in turn means we can avoid cracks in rasterization without
    // using fixed-point arithmetic.
    // See http://mathworld.wolfram.com/MatrixInverse.html

    template <typename scalar_t>
    __device__ void ComputeUnnormalizedMatrixInverse(
        const scalar_t a11, const scalar_t a12,
        const scalar_t a13, const scalar_t a21,
        const scalar_t a22, const scalar_t a23,
        const scalar_t a31, const scalar_t a32,
        const scalar_t a33, scalar_t m_inv[9])
    {
        m_inv[0] = a22 * a33 - a32 * a23;
        m_inv[1] = a13 * a32 - a33 * a12;
        m_inv[2] = a12 * a23 - a22 * a13;
        m_inv[3] = a23 * a31 - a33 * a21;
        m_inv[4] = a11 * a33 - a31 * a13;
        m_inv[5] = a13 * a21 - a23 * a11;
        m_inv[6] = a21 * a32 - a31 * a22;
        m_inv[7] = a12 * a31 - a32 * a11;
        m_inv[8] = a11 * a22 - a21 * a12;

        // The first column of the unnormalized M^-1 contains intermediate values for
        // det(M).
        const scalar_t det = a11 * m_inv[0] + a12 * m_inv[3] + a13 * m_inv[6];

        // Transfer the sign of the determinant.
        if (det < 0.0f)
        {
            for (int32_t i = 0; i < 9; ++i)
            {
                m_inv[i] = -m_inv[i];
            }
        }
    }

    // Computes the edge functions from M^-1 as described by Olano and Greer,
    // "Triangle Scan Conversion using 2D Homogeneous Coordinates."
    //
    // This function combines equations (3) and (4). It first computes
    // [a b c] = u_i * M^-1, where u_0 = [1 0 0], u_1 = [0 1 0], etc.,
    // then computes edge_i = aX + bY + c

    template <typename scalar_t>
    __device__ void ComputeEdgeFunctions(const scalar_t px, const scalar_t py,
        const scalar_t m_inv[9], scalar_t values[3])
    {
        for (int32_t i = 0; i < 3; ++i)
        {
            const scalar_t a = m_inv[3 * i + 0];
            const scalar_t b = m_inv[3 * i + 1];
            const scalar_t c = m_inv[3 * i + 2];

            values[i] = a * px + b * py + c;
        }
    }

    // Determines whether the point p lies inside a front-facing triangle.
    // Counts pixels exactly on an edge as inside the triangle, as long as the
    // triangle is not degenerate. Degenerate (zero-area) triangles always fail the
    // inside test.
    template <typename scalar_t>
    __device__ __forceinline__ bool PixelIsInsideTriangle(const scalar_t edge_values[3])
    {
        // Check that the edge values are all non-negative and that at least one is
        // positive (triangle is non-degenerate).
        return (edge_values[0] >= 0 && edge_values[1] >= 0 && edge_values[2] >= 0) &&
            (edge_values[0] > 0 || edge_values[1] > 0 || edge_values[2] > 0);
    }

    template <typename scalar_t>
    __global__ void RasterizeTrianglesForwardCudaKernel(
        const scalar_t* vertices, const int64_t* triangles,
        int64_t triangle_count, int32_t image_width, int32_t image_height,
        scalar_t* barycentric_coordinates, int64_t* triangle_ids,
        scalar_t* z_buffer, int32_t* locks)
    {
        const int64_t triangle_id = blockIdx.x * blockDim.x + threadIdx.x;

        if (triangle_id >= triangle_count)
        {
            return;
        }

        const scalar_t half_image_width = 0.5f * image_width;
        const scalar_t half_image_height = 0.5f * image_height;
        double unnormalized_matrix_inverse[9];
        double b_over_w[3];

        const int64_t v0_x_id = 4 * triangles[3 * triangle_id];
        const int64_t v1_x_id = 4 * triangles[3 * triangle_id + 1];
        const int64_t v2_x_id = 4 * triangles[3 * triangle_id + 2];

        const scalar_t v0w = vertices[v0_x_id + 3];
        const scalar_t v1w = vertices[v1_x_id + 3];
        const scalar_t v2w = vertices[v2_x_id + 3];
        // Early exit: if all w < 0, triangle is entirely behind the eye.
        if (v0w < 0 && v1w < 0 && v2w < 0)
        {
            return;
        }

        const scalar_t v0x = vertices[v0_x_id];
        const scalar_t v0y = vertices[v0_x_id + 1];
        const scalar_t v1x = vertices[v1_x_id];
        const scalar_t v1y = vertices[v1_x_id + 1];
        const scalar_t v2x = vertices[v2_x_id];
        const scalar_t v2y = vertices[v2_x_id + 1];

        // The nondeterminacy of GPU device in single precision may lead some pixel 
        // to be missing when a pixel is on the boundary of two triangles, so we use 
        // double precision to check the location of a pixel.
        ComputeUnnormalizedMatrixInverse((double)v0x, (double)v1x, (double)v2x,
            (double)v0y, (double)v1y, (double)v2y,
            (double)v0w, (double)v1w, (double)v2w,
            unnormalized_matrix_inverse);

        // Initialize the bounding box to the entire screen.
        int32_t left = 0, right = image_width, bottom = 0, top = image_height;
        // If the triangle is entirely inside the screen, project the vertices to
        // pixel coordinates and find the triangle bounding box enlarged to the
        // nearest integer and clamped to the image boundaries.
        if (v0w > 0 && v1w > 0 && v2w > 0)
        {
            const scalar_t p0x = (v0x / v0w + 1.0f) * half_image_width;
            const scalar_t p1x = (v1x / v1w + 1.0f) * half_image_width;
            const scalar_t p2x = (v2x / v2w + 1.0f) * half_image_width;
            const scalar_t p0y = (v0y / v0w + 1.0f) * half_image_height;
            const scalar_t p1y = (v1y / v1w + 1.0f) * half_image_height;
            const scalar_t p2y = (v2y / v2w + 1.0f) * half_image_height;

            left = ClampedIntegerMin(p0x, p1x, p2x, 0, image_width);
            right = ClampedIntegerMax(p0x, p1x, p2x, 0, image_width);
            bottom = ClampedIntegerMin(p0y, p1y, p2y, 0, image_height);
            top = ClampedIntegerMax(p0y, p1y, p2y, 0, image_height);
        }

        // Iterate over each pixel in the bounding box.
        for (int32_t iy = bottom; iy < top; ++iy)
        {
            for (int32_t ix = left; ix < right; ++ix)
            {
                const scalar_t px = ((ix + 0.5f) / half_image_width) - 1.0f;
                const scalar_t py = ((iy + 0.5f) / half_image_height) - 1.0f;
                const int32_t pixel_idx = iy * image_width + ix;

                ComputeEdgeFunctions((double)px, (double)py, unnormalized_matrix_inverse, b_over_w);

                if (!PixelIsInsideTriangle(b_over_w))
                {
                    continue;
                }

                const scalar_t one_over_w = scalar_t(b_over_w[0] + b_over_w[1] + b_over_w[2]);
                const scalar_t b0 = scalar_t(b_over_w[0] / one_over_w);
                const scalar_t b1 = scalar_t(b_over_w[1] / one_over_w);
                const scalar_t b2 = scalar_t(b_over_w[2] / one_over_w);

                const scalar_t v0z = vertices[v0_x_id + 2];
                const scalar_t v1z = vertices[v1_x_id + 2];
                const scalar_t v2z = vertices[v2_x_id + 2];
                // Since we computed an unnormalized w above, we need to recompute
                // a properly scaled clip-space w value and then divide clip-space z
                // by that.
                const scalar_t clip_z = b0 * v0z + b1 * v1z + b2 * v2z;
                const scalar_t clip_w = b0 * v0w + b1 * v1w + b2 * v2w;
                const scalar_t z = clip_z / clip_w;

                // Skip the pixel if it is farther than the current z-buffer pixel or
                // beyond the near or far clipping plane.
                if (z < -1.0 || z > 1.0) // || z > z_buffer[pixel_idx]
                {
                    continue;
                }

                // write z_buffer, triangle_ids and barycentric_coordinates by using cuda threads lock
                // reference: https://stackoverflow.com/questions/21341495/cuda-mutex-and-atomiccas
                int32_t* mutex = locks + pixel_idx;
                bool isSet = false;
                do
                {
                    if (isSet = atomicCAS(mutex, 0, 1) == 0)
                    {
                        if (z <= z_buffer[pixel_idx])
                        {
                            z_buffer[pixel_idx] = z;
                            triangle_ids[pixel_idx] = triangle_id;
                            barycentric_coordinates[3 * pixel_idx + 0] = b0;
                            barycentric_coordinates[3 * pixel_idx + 1] = b1;
                            barycentric_coordinates[3 * pixel_idx + 2] = b2;
                        }
                    }
                    if (isSet)
                    {
                        atomicExch(mutex, 0);
                        __threadfence();
                    }
                } while (!isSet);
                /* original
                if (z < z_buffer[pixel_idx])
                {
                    z_buffer[pixel_idx] = z;
                    triangle_ids[pixel_idx] = triangle_id;
                    barycentric_coordinates[3 * pixel_idx + 0] = b0;
                    barycentric_coordinates[3 * pixel_idx + 1] = b1;
                    barycentric_coordinates[3 * pixel_idx + 2] = b2;
                }
                */
            }
        }
    }

    void RasterizeTrianglesForwardCuda(
        at::Tensor vertices, at::Tensor triangles,
        int32_t image_width, int32_t image_height,
        torch::Tensor barycentric, torch::Tensor triangle_ids, torch::Tensor z_buffer)
    {
        const int64_t triangle_count = triangles.size(0);
        const int threads = 512;
        const dim3 blocks((triangle_count - 1) / threads + 1);

        int32_t* locks = NULL; // pixel locks
        cudaMalloc((void**)&locks, image_width * image_height * sizeof(int32_t));
        cudaMemset(locks, 0, image_width * image_height * sizeof(int32_t));

        AT_DISPATCH_FLOATING_TYPES(vertices.type(), "RasterizeTrianglesForwardCuda", ([&] {
            RasterizeTrianglesForwardCudaKernel<scalar_t> << <blocks, threads >> > (
                vertices.data<scalar_t>(),
                triangles.data<int64_t>(),
                triangle_count,
                image_width,
                image_height,
                barycentric.data<scalar_t>(),
                triangle_ids.data<int64_t>(),
                z_buffer.data<scalar_t>(),
                locks);
        }));

        cudaFree(locks);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            printf("Error in RasterizeTrianglesForwardCuda: %s\n", cudaGetErrorString(err));
    }

    template <typename scalar_t>
    __global__ void RasterizeTrianglesBackwardCudaKernel(
        const scalar_t* vertices, const int64_t* triangles,
        const scalar_t* barycentric_coordinates,
        const int64_t* triangle_ids,
        const scalar_t* df_dbarycentric_coordinates,
        int32_t image_width, int32_t image_height,
        scalar_t* df_dvertices)
    {
        const int32_t pixel_id = blockIdx.x * blockDim.x + threadIdx.x;
        if (pixel_id >= image_width * image_height)
        {
            return;
        }
        // We first loop over each pixel in the output image, and compute
        // dbarycentric_coordinate[0,1,2]/dvertex[0x, 0y, 1x, 1y, 2x, 2y].
        // Next we compute each value above's contribution to
        // df/dvertices, building up that matrix as the output of this iteration.

        // b0, b1, and b2 are the three barycentric coordinate values
        // rendered at pixel pixel_id.
        const scalar_t b0 = barycentric_coordinates[3 * pixel_id];
        const scalar_t b1 = barycentric_coordinates[3 * pixel_id + 1];
        const scalar_t b2 = barycentric_coordinates[3 * pixel_id + 2];

        if (b0 + b1 + b2 < kDegenerateBarycentricCoordinatesCutoff)
        {
            return;
        }

        const scalar_t df_db0 = df_dbarycentric_coordinates[3 * pixel_id];
        const scalar_t df_db1 = df_dbarycentric_coordinates[3 * pixel_id + 1];
        const scalar_t df_db2 = df_dbarycentric_coordinates[3 * pixel_id + 2];

        const int64_t triangle_at_current_pixel = triangle_ids[pixel_id];
        const int64_t* vertices_at_current_pixel =
            &triangles[3 * triangle_at_current_pixel];

        // Extract vertex indices for the current triangle.
        const int64_t v0_id = 4 * vertices_at_current_pixel[0];
        const int64_t v1_id = 4 * vertices_at_current_pixel[1];
        const int64_t v2_id = 4 * vertices_at_current_pixel[2];

        // Extract x,y,w components of the vertices' clip space coordinates.
        const scalar_t x0 = vertices[v0_id];
        const scalar_t y0 = vertices[v0_id + 1];
        const scalar_t w0 = vertices[v0_id + 3];
        const scalar_t x1 = vertices[v1_id];
        const scalar_t y1 = vertices[v1_id + 1];
        const scalar_t w1 = vertices[v1_id + 3];
        const scalar_t x2 = vertices[v2_id];
        const scalar_t y2 = vertices[v2_id + 1];
        const scalar_t w2 = vertices[v2_id + 3];

        // Compute pixel's NDC-s.
        const int32_t ix = pixel_id % image_width;
        const int32_t iy = pixel_id / image_width;
        const scalar_t px = 2 * (ix + 0.5f) / image_width - 1.0f;
        const scalar_t py = 2 * (iy + 0.5f) / image_height - 1.0f;

        // Baricentric gradients wrt each vertex coordinate share a common factor.
        const scalar_t db0_dx = py * (w1 - w2) - (y1 - y2);
        const scalar_t db1_dx = py * (w2 - w0) - (y2 - y0);
        const scalar_t db2_dx = -(db0_dx + db1_dx);
        const scalar_t db0_dy = (x1 - x2) - px * (w1 - w2);
        const scalar_t db1_dy = (x2 - x0) - px * (w2 - w0);
        const scalar_t db2_dy = -(db0_dy + db1_dy);
        const scalar_t db0_dw = px * (y1 - y2) - py * (x1 - x2);
        const scalar_t db1_dw = px * (y2 - y0) - py * (x2 - x0);
        const scalar_t db2_dw = -(db0_dw + db1_dw);

        // Combine them with chain rule.
        const scalar_t df_dx = df_db0 * db0_dx + df_db1 * db1_dx + df_db2 * db2_dx;
        const scalar_t df_dy = df_db0 * db0_dy + df_db1 * db1_dy + df_db2 * db2_dy;
        const scalar_t df_dw = df_db0 * db0_dw + df_db1 * db1_dw + df_db2 * db2_dw;

        // Values of edge equations and inverse w at the current pixel.
        const scalar_t edge0_over_w = x2 * db0_dx + y2 * db0_dy + w2 * db0_dw;
        const scalar_t edge1_over_w = x2 * db1_dx + y2 * db1_dy + w2 * db1_dw;
        const scalar_t edge2_over_w = x1 * db2_dx + y1 * db2_dy + w1 * db2_dw;
        const scalar_t w_inv = edge0_over_w + edge1_over_w + edge2_over_w;

        // All gradients share a common denominator.
        const scalar_t w_sqr = 1 / (w_inv * w_inv);

        // Gradients wrt each vertex share a common factor.
        const scalar_t edge0 = w_sqr * edge0_over_w;
        const scalar_t edge1 = w_sqr * edge1_over_w;
        const scalar_t edge2 = w_sqr * edge2_over_w;

        atomicAdd(&df_dvertices[v0_id + 0], edge0 * df_dx);
        atomicAdd(&df_dvertices[v0_id + 1], edge0 * df_dy);
        atomicAdd(&df_dvertices[v0_id + 3], edge0 * df_dw);
        atomicAdd(&df_dvertices[v1_id + 0], edge1 * df_dx);
        atomicAdd(&df_dvertices[v1_id + 1], edge1 * df_dy);
        atomicAdd(&df_dvertices[v1_id + 3], edge1 * df_dw);
        atomicAdd(&df_dvertices[v2_id + 0], edge2 * df_dx);
        atomicAdd(&df_dvertices[v2_id + 1], edge2 * df_dy);
        atomicAdd(&df_dvertices[v2_id + 3], edge2 * df_dw);
    }

    void RasterizeTrianglesBackwardCuda(
        at::Tensor vertices,                    // FloatTensor
        at::Tensor triangles,                   // LongTensor
        at::Tensor barycentric_coordinates,     // FloatTensor
        at::Tensor triangle_ids,                // LongTensor
        at::Tensor df_dbarycentric_coordinates, // FloatTensor
        int32_t image_width,
        int32_t image_height,
        at::Tensor df_dvertices)
    {
        const int64_t pixel_count = image_width * image_height;
        const int threads = 512;
        const dim3 blocks((pixel_count - 1) / threads + 1);

        AT_DISPATCH_FLOATING_TYPES(vertices.type(), "RasterizeTrianglesBackwardCuda", ([&] {
            RasterizeTrianglesBackwardCudaKernel<scalar_t> << <blocks, threads >> > (
                vertices.data<scalar_t>(),
                triangles.data<int64_t>(),
                barycentric_coordinates.data<scalar_t>(),
                triangle_ids.data<int64_t>(),
                df_dbarycentric_coordinates.data<scalar_t>(),
                image_width,
                image_height,
                df_dvertices.data<scalar_t>());
        }));

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            printf("Error in RasterizeTrianglesBackwardCuda: %s\n", cudaGetErrorString(err));
    }

}// namespace pytorch_mesh_renderer
