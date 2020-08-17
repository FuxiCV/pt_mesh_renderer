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

# Modifications: Pytorch implementation of rasterizer
# Copyright 2020 Netease Fuxi AI LAB
# SPDX-License-Identifier: Apache-2.0

import torch
from .RasterizeTriangles import rasterize_triangles
from .camera_utils import transform_homogeneous


def rasterize(world_space_vertices, attributes, triangles, camera_matrices,
              image_width, image_height, background_value):
    """Rasterizes a mesh and computes interpolated vertex attributes.

    Applies projection matrices and then calls rasterize_clip_space().

    Args:
        world_space_vertices: 3-D float32 tensor of xyz positions with shape
            [batch_size, vertex_count, 3].
        attributes: 3-D float32 tensor with shape [batch_size, vertex_count,
            attribute_count]. Each vertex attribute is interpolated across the
            triangle using barycentric interpolation.
        triangles: 2-D int32 tensor with shape [triangle_count, 3]. Each triplet
            should contain vertex indices describing a triangle such that the
            triangle's normal points toward the viewer if the forward order of the
            triplet defines a clockwise winding of the vertices. Gradients with
            respect to this tensor are not available.
        camera_matrices: 3-D float tensor with shape [batch_size, 4, 4] containing
            model-view-perspective projection matrices.
        image_width: int specifying desired output image width in pixels.
        image_height: int specifying desired output image height in pixels.
        background_value: a 1-D float32 tensor with shape [attribute_count]. Pixels
            that lie outside all triangles take this value.

    Returns:
        A 4-D float32 tensor with shape [batch_size, image_height, image_width,
        attribute_count], containing the interpolated vertex attributes at
        each pixel.

    Raises:
        ValueError: An invalid argument to the method is detected.
    """
    print("Warning: rasterize() has been deprecated!")
    clip_space_vertices = transform_homogeneous(camera_matrices, world_space_vertices)
    return rasterize_clip_space(clip_space_vertices, attributes, triangles,
                                image_width, image_height, background_value)


def rasterize_clip_space(clip_space_vertices, attributes, triangles,
                         image_width, image_height, background_value):
    """Rasterizes the input mesh expressed in clip-space (xyzw) coordinates.

    Interpolates vertex attributes using perspective-correct interpolation and
    clips triangles that lie outside the viewing frustum.

    Args:
        clip_space_vertices: 3-D float32 tensor of homogenous vertices (xyzw) with
            shape [batch_size, vertex_count, 4].
        attributes: 3-D float32 tensor with shape [batch_size, vertex_count,
            attribute_count]. Each vertex attribute is interpolated across the
            triangle using barycentric interpolation.
        triangles: 2-D int32 tensor with shape [triangle_count, 3]. Each triplet
            should contain vertex indices describing a triangle such that the
            triangle's normal points toward the viewer if the forward order of the
            triplet defines a clockwise winding of the vertices. Gradients with
            respect to this tensor are not available.
        image_width: int specifying desired output image width in pixels.
        image_height: int specifying desired output image height in pixels.
        background_value: a 1-D float32 tensor with shape [attribute_count]. Pixels
            that lie outside all triangles take this value.

    Returns:
        A 4-D float32 tensor with shape [batch_size, image_height, image_width,
        attribute_count], containing the interpolated vertex attributes at
        each pixel.

    Raises:
        ValueError: An invalid argument to the method is detected.
    """
    if not image_width > 0:
        raise ValueError('Image width must be > 0.')
    if not image_height > 0:
        raise ValueError('Image height must be > 0.')
    if len(clip_space_vertices.shape) != 3:
        raise ValueError('The vertex buffer must be 3D.')

    vertex_count = clip_space_vertices.shape[1]

    batch_size = clip_space_vertices.shape[0]

    # Original type is tf.TensorArray
    per_image_barycentric_coordinates = []

    per_image_vertex_ids = []

    for b in range(batch_size):
        barycentric_coords, triangle_ids, z_buffer = \
            rasterize_triangles(clip_space_vertices[b, :, :], triangles,
                                image_width, image_height)
        per_image_barycentric_coordinates.append(barycentric_coords.view(-1, 3))
        vertex_ids = triangles[triangle_ids.view(-1)]
        reindexed_ids = vertex_ids + b * clip_space_vertices.shape[1]
        per_image_vertex_ids.append(reindexed_ids)

    barycentric_coordinates = torch.cat(per_image_barycentric_coordinates, dim=0)
    vertex_ids = torch.cat(per_image_vertex_ids, dim=0)

    # Indexes with each pixel's clip-space triangle's extrema (the pixel's
    # 'corner points') ids to get the relevant properties for deferred shading.
    flattened_vertex_attributes = attributes.view([batch_size * vertex_count, -1])

    corner_attributes = flattened_vertex_attributes[vertex_ids, :]

    # Computes the pixel attributes by interpolating the known attributes at the
    # corner points of the triangle interpolated with the barycentric coordinates.
    weighted_vertex_attributes = corner_attributes * barycentric_coordinates.unsqueeze(2)

    summed_attributes = weighted_vertex_attributes.sum(dim=1)
    attribute_images = summed_attributes.view(batch_size, image_height, image_width, -1)

    # Barycentric coordinates should approximately sum to one where there is
    # rendered geometry, but be exactly zero where there is not.
    alphas = (2.0 * barycentric_coordinates).sum(dim=1).clamp(0, 1)
    alphas = alphas.view(batch_size, image_height, image_width, 1)

    background_value = torch.FloatTensor(background_value).expand_as(attribute_images).to(clip_space_vertices.device)
    attributes_with_background = (alphas * attribute_images + (1.0 - alphas) * background_value)

    return attributes_with_background
