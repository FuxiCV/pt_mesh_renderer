# Copyright 2017 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modifications: this file implements a pytorch interface to c++ codes
# Copyright 2020 Netease Fuxi AI LAB
# SPDX-License-Identifier: Apache-2.0

from torch.autograd import Function
import pt_mesh_renderer.kernels.rasterize_triangles as rasterize_triangles_kernels
try:
    import pt_mesh_renderer.kernels.rasterize_triangles_cuda as rasterize_triangles_kernels_cuda
except Exception:
    print("Cannot import cuda rasterizer, renderer is running in CPU mode.")


class RasterizeTriangles(Function):
    @staticmethod
    def forward(ctx, vertices, triangles, image_width, image_height):
        """Rasterizes the single input mesh expressed in clip-space (xyzw) coordinates.

        Args:
            vertices: 2-D float32 tensor of homogenous vertices (xyzw) with
                shape [vertex_count, 4].
            triangles: 2-D int32 tensor with shape [triangle_count, 3]. Each triplet
                should contain vertex indices describing a triangle such that the
                triangle's normal points toward the viewer if the forward order of the
                triplet defines a clockwise winding of the vertices. Gradients with
                respect to this tensor are not available.
            image_width: int specifying desired output image width in pixels.
            image_height: int specifying desired output image height in pixels.
        Returns:
            barycentric_coordinates: 3-D tensor with shape [image_height, image_width, 3]
                containing the rendered barycentric coordinate triplet per pixel, before
                perspective correction. The triplet is the zero vector if the pixel is outside
                the mesh boundary. For valid pixels, the ordering of the coordinates
                corresponds to the ordering in triangles.
            triangle_ids: 2-D tensor with shape [image_height, image_width]. Contains the
                triangle id value for each pixel in the output image. For pixels within the
                mesh, this is the integer value in the range [0, num_vertices] from triangles.
                For vertices outside the mesh this is 0; 0 can either indicate belonging to
                triangle 0, or being outside the mesh. This ensures all returned triangle ids
                will validly index into the vertex array, enabling the use of torch.index_select
                (instead of tf.gather) with indices from this tensor. The barycentric coordinates 
                can be used to determine pixel validity instead.
            z_buffer: 2-D tensor with shape [image_height, image_width]. Contains the Z
                coordinate in Normalized Device Coordinates for each pixel occupied by a
                triangle.
        """
        # project mesh to image
        if vertices.is_cuda:
            forward_function = rasterize_triangles_kernels_cuda.forward_rasterize_triangles_cuda
        else:
            forward_function = rasterize_triangles_kernels.forward_rasterize_triangles
        barycentric, triangle_ids, z_buffer = forward_function(vertices, triangles, image_width, image_height)

        # only barycentric needs grad
        ctx.mark_non_differentiable(triangle_ids, z_buffer)

        # save variables
        ctx.save_for_backward(vertices, barycentric, triangle_ids)
        ctx.triangles = triangles
        ctx.image_size = [image_width, image_height]
        return barycentric, triangle_ids, z_buffer

    @staticmethod
    def backward(ctx, grad_barycentric, grad_triangle_ids, grad_z_buffer):
        # get variables
        vertices, barycentric, triangle_ids = ctx.saved_tensors
        triangles = ctx.triangles
        image_width, image_height = ctx.image_size

        # compute grad from image to mesh vertices
        if vertices.is_cuda:
            backward_function = rasterize_triangles_kernels_cuda.backward_rasterize_triangles_cuda
        else:
            backward_function = rasterize_triangles_kernels.backward_rasterize_triangles
        grad_vertices = backward_function(
            vertices, triangles, barycentric, triangle_ids, grad_barycentric, image_width, image_height
        )

        return grad_vertices[0], None, None, None


rasterize_triangles = RasterizeTriangles.apply
