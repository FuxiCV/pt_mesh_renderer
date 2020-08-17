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

# Modifications: Pytorch implementation of camera utils
# Copyright 2020 Netease Fuxi AI LAB
# SPDX-License-Identifier: Apache-2.0

import math
import torch


def euler_matrices(angles):
    """Computes a XYZ Tait-Bryan (improper Euler angle) rotation.

    Returns 4x4 matrices for convenient multiplication with other transformations.

    Args:
        angles: a [batch_size, 3] tensor containing X, Y, and Z angles in radians.

    Returns:
        a [batch_size, 4, 4] tensor of matrices.
    """
    s = torch.sin(angles)
    c = torch.cos(angles)
    # Rename variables for readability in the matrix definition below.
    c0, c1, c2 = (c[:, 0], c[:, 1], c[:, 2])
    s0, s1, s2 = (s[:, 0], s[:, 1], s[:, 2])

    zeros = torch.zeros_like(s[:, 0])
    ones = torch.ones_like(s[:, 0])

    # pyformat: disable
    flattened = torch.cat(
        [
            c2 * c1, c2 * s1 * s0 - c0 * s2, s2 * s0 + c2 * c0 * s1, zeros,
            c1 * s2, c2 * c0 + s2 * s1 * s0, c0 * s2 * s1 - c2 * s0, zeros,
            -s1, c1 * s0, c1 * c0, zeros,
            zeros, zeros, zeros, ones
        ],
        dim=0)
    # pyformat: enable
    reshaped = flattened.view(4, 4, -1)
    return reshaped.transpose(2, 0).transpose(2, 1)  # TODO


def perspective(aspect_ratio, fov_y, near_clip, far_clip):
    """Computes perspective transformation matrices.

    Functionality mimes gluPerspective (third_party/GL/glu/include/GLU/glu.h).

    Args:
        aspect_ratio: float value specifying the image aspect ratio (width/height).
        fov_y: 1-D float32 Tensor with shape [batch_size] specifying output vertical
            field of views in degrees.
        near_clip: 1-D float32 Tensor with shape [batch_size] specifying near
            clipping plane distance.
        far_clip: 1-D float32 Tensor with shape [batch_size] specifying far clipping
            plane distance.

    Returns:
        A [batch_size, 4, 4] float tensor that maps from right-handed points in eye
        space to left-handed points in clip space.
    """
    # The multiplication of fov_y by pi/360.0 simultaneously converts to radians
    # and adds the half-angle factor of .5.
    focal_lengths_y = 1.0 / torch.tan(fov_y * (math.pi / 360.0))
    depth_range = far_clip - near_clip
    p_22 = -(far_clip + near_clip) / depth_range
    p_23 = -2.0 * (far_clip * near_clip / depth_range)

    zeros = torch.zeros_like(p_23)
    # pyformat: disable
    perspective_transform = torch.cat(
        [
            focal_lengths_y / aspect_ratio, zeros, zeros, zeros,
            zeros, focal_lengths_y, zeros, zeros,
            zeros, zeros, p_22, p_23,
            zeros, zeros, -torch.ones_like(p_23), zeros
        ], dim=0)
    # pyformat: enable
    perspective_transform = perspective_transform.view([4, 4, -1])
    return perspective_transform.transpose(2, 0).transpose(2, 1)  # transpose([2, 0, 1])


def look_at(eye, center, world_up):
    """Computes camera viewing matrices.

    Functionality mimes gluLookAt (third_party/GL/glu/include/GLU/glu.h).

    Args:
        eye: 2-D float32 tensor with shape [batch_size, 3] containing the XYZ world
            space position of the camera.
        center: 2-D float32 tensor with shape [batch_size, 3] containing a position
            along the center of the camera's gaze.
        world_up: 2-D float32 tensor with shape [batch_size, 3] specifying the
            world's up direction; the output camera will have no tilt with respect
            to this direction.

    Returns:
        A [batch_size, 4, 4] float tensor containing a right-handed camera
        extrinsics matrix that maps points from world space to points in eye space.
    """
    batch_size = center.shape[0]
    device = eye.device
    vector_degeneracy_cutoff = 1e-6
    forward = center - eye
    forward_norm = forward.norm(p='fro', dim=1, keepdim=True)
    if (forward_norm.data <= vector_degeneracy_cutoff).any():
        raise 'Camera matrix is degenerate because eye and center are close.'

    forward = forward / forward_norm

    to_side = torch.cross(forward, world_up)
    to_side_norm = to_side.norm(p='fro', dim=1, keepdim=True)
    if (to_side_norm.data <= vector_degeneracy_cutoff).any():
        raise 'Camera matrix is degenerate because up and gaze are close or because up is degenerate.'

    to_side = to_side / to_side_norm
    cam_up = torch.cross(to_side, forward)

    w_column = torch.FloatTensor(
        batch_size * [[0., 0., 0., 1.]]).view([batch_size, 4, 1]).to(device)  # [batch_size, 4, 1]

    view_rotation = torch.stack(
        [to_side, cam_up, -forward, torch.zeros_like(to_side).float()],
        dim=1)  # [batch_size, 4, 3] matrix
    view_rotation = torch.cat(
        [view_rotation, w_column], dim=2)  # [batch_size, 4, 4]

    identity_batch = torch.eye(3).view(1, 3, 3).expand([batch_size, -1, -1]).to(device)
    view_translation = torch.cat([identity_batch, -eye.unsqueeze(dim=2)], dim=2)
    view_translation = torch.cat([view_translation, w_column.view([batch_size, 1, 4])], dim=1)
    camera_matrices = torch.matmul(view_rotation, view_translation)
    return camera_matrices


def transform_homogeneous(matrices, vertices):
    """Applies batched 4x4 homogenous matrix transformations to 3-D vertices.

    The vertices are input and output as as row-major, but are interpreted as
    column vectors multiplied on the right-hand side of the matrices. More
    explicitly, this function computes (MV^T)^T.
    Vertices are assumed to be xyz, and are extended to xyzw with w=1.

    Args:
        matrices: a [batch_size, 4, 4] tensor of matrices.
        vertices: a [batch_size, N, 3] tensor of xyz vertices.

    Returns:
        a [batch_size, N, 4] tensor of xyzw vertices.

    Raises:
        ValueError: if matrices or vertices have the wrong number of dimensions.
    """
    if len(matrices.shape) != 3:
        raise ValueError(
            'matrices must have 3 dimensions (missing batch dimension?)')
    if len(vertices.shape) != 3:
        raise ValueError(
            'vertices must have 3 dimensions (missing batch dimension?)')
    homogeneous_coord = torch.ones([vertices.shape[0], vertices.shape[1], 1]).to(vertices.device)
    vertices_homogeneous = torch.cat([vertices, homogeneous_coord], 2)

    return torch.matmul(vertices_homogeneous, matrices.transpose(1, 2))


def normalize_homogeneous(vertices_homogeneous, image_width, image_height):
    """
    Transform the homogeneous coords of vertices to the image space
    """
    x = vertices_homogeneous[:, :, 0:1]
    y = vertices_homogeneous[:, :, 1:2]
    w = vertices_homogeneous[:, :, 3:4]
    x_n = (x / w + 1.0) / 2 * image_width
    y_n = image_height - (y / w + 1.0) / 2 * image_height
    return torch.cat([x_n, y_n], dim=2)
