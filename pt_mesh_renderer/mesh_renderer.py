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

# Modifications: Pytorch implementation of mesh_renderer
# Copyright 2020 Netease Fuxi AI LAB
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from .camera_utils import look_at, perspective, transform_homogeneous, normalize_homogeneous
from .rasterizer import rasterize_clip_space


def phong_shader(normals,
                 alphas,
                 pixel_positions,
                 light_positions,
                 light_intensities,
                 diffuse_colors=None,
                 camera_position=None,
                 specular_colors=None,
                 shininess_coefficients=None,
                 ambient_color=None):
    """Computes pixelwise lighting from rasterized buffers with the Phong model.

    Args:
        normals: a 4D float32 tensor with shape [batch_size, image_height,
            image_width, 3]. The inner dimension is the world space XYZ normal for
            the corresponding pixel. Should be already normalized.
        alphas: a 3D float32 tensor with shape [batch_size, image_height,
            image_width]. The inner dimension is the alpha value (transparency)
            for the corresponding pixel.
        pixel_positions: a 4D float32 tensor with shape [batch_size, image_height,
            image_width, 3]. The inner dimension is the world space XYZ position for
            the corresponding pixel.
        light_positions: a 3D tensor with shape [batch_size, light_count, 3]. The
            XYZ position of each light in the scene. In the same coordinate space as
            pixel_positions.
        light_intensities: a 3D tensor with shape [batch_size, light_count, 3]. The
            RGB intensity values for each light. Intensities may be above one.
        diffuse_colors: a 4D float32 tensor with shape [batch_size, image_height,
            image_width, 3]. The inner dimension is the diffuse RGB coefficients at
            a pixel in the range [0, 1].
        camera_position: a 1D tensor with shape [batch_size, 3]. The XYZ camera
            position in the scene. If supplied, specular reflections will be
            computed. If not supplied, specular_colors and shininess_coefficients
            are expected to be None. In the same coordinate space as
            pixel_positions.
        specular_colors: a 4D float32 tensor with shape [batch_size, image_height,
            image_width, 3]. The inner dimension is the specular RGB coefficients at
            a pixel in the range [0, 1]. If None, assumed to be zeros.
        shininess_coefficients: A 3D float32 tensor that is broadcasted to shape
            [batch_size, image_height, image_width]. The inner dimension is the
            shininess coefficient for the object at a pixel. Dimensions that are
            constant can be given length 1, so [batch_size, 1, 1] and [1, 1, 1] are
            also valid input shapes.
        ambient_color: a 2D tensor with shape [batch_size, 3]. The RGB ambient
            color, which is added to each pixel before tone mapping. If None, it is
            assumed to be zeros.
    Returns:
        A 4D float32 tensor of shape [batch_size, image_height, image_width, 4]
        containing the lit RGBA color values for each image at each pixel. Colors
        are in the range [0,1].

    Raises:
        ValueError: An invalid argument to the method is detected.
    """
    batch_size, image_height, image_width, _ = normals.shape
    light_count = light_positions.shape[1]
    pixel_count = image_height * image_width

    # Reshape all values to easily do pixelwise computations:
    normals = normals.view(batch_size, -1, 3)
    alphas = alphas.view(batch_size, -1, 1)
    diffuse_colors = diffuse_colors.view(batch_size, -1, 3)

    if camera_position is not None:
        specular_colors = specular_colors.view(batch_size, -1, 3)

    # Ambient component
    output_colors = torch.zeros([batch_size, image_height * image_width, 3], device=normals.device)
    if ambient_color is not None:
        ambient_reshaped = ambient_color.unsqueeze(1)
        output_colors = output_colors + ambient_reshaped * diffuse_colors

    # Diffuse component
    pixel_positions = pixel_positions.view(batch_size, -1, 3)
    per_light_pixel_positions = torch.stack(
        [pixel_positions] * light_count,
        dim=1)  # [batch_size, light_count, pixel_count, 3]
    directions_to_lights = nn.functional.normalize(
        light_positions.unsqueeze(2) - per_light_pixel_positions,
        p=2, dim=3)  # [batch_size, light_count, pixel_count, 3]

    # The specular component should only contribute when the light and normal
    # face one another (i.e. the dot product is nonnegative):
    # [batch_size, light_count, pixel_count]
    normals_dot_lights = (normals.unsqueeze(1) * directions_to_lights).sum(dim=3).clamp(0, 1)

    diffuse_output = diffuse_colors.unsqueeze(1) * normals_dot_lights.unsqueeze(3) * light_intensities.unsqueeze(2)
    diffuse_output = diffuse_output.sum(dim=1)  # [batch_size, pixel_count, 3]
    output_colors = output_colors + diffuse_output

    # Specular component
    if camera_position is not None:
        camera_position = camera_position.view(batch_size, 1, 3)
        mirror_reflection_direction = nn.functional.normalize(
            2.0 * normals_dot_lights.unsqueeze(3) * normals.unsqueeze(1) - directions_to_lights,
            p=2, dim=3)
        direction_to_camera = \
            nn.functional.normalize(camera_position - pixel_positions, p=2, dim=2)
        reflection_direction_dot_camera_direction = \
            (direction_to_camera.unsqueeze(1) * mirror_reflection_direction).sum(dim=3)

        # The specular component should only contribute when the reflection is
        # external:
        reflection_direction_dot_camera_direction = \
            nn.functional.normalize(reflection_direction_dot_camera_direction, p=2, dim=2).clamp(0, 1)

        # The specular component should also only contribute when the diffuse
        # component contributes:
        reflection_direction_dot_camera_direction[normals_dot_lights != 0.0] = 0
        # Reshape to support broadcasting the shininess coefficient, which rarely
        # varies per-vertex:
        reflection_direction_dot_camera_direction = \
            reflection_direction_dot_camera_direction.view(
                batch_size, light_count, image_height, image_width)
        shininess_coefficients = shininess_coefficients.unsqueeze(1)
        specularity = (reflection_direction_dot_camera_direction **
                       shininess_coefficients).view(batch_size, light_count, pixel_count, 1)

        specular_output = specular_colors.unsqueeze(1) * specularity * light_intensities.unsqueeze(2)
        specular_output = specular_output.sum(dim=1)
        output_colors = output_colors + specular_output

    rgb_images = output_colors.view(batch_size, image_height, image_width, 3)
    alpha_images = alphas.view(batch_size, image_height, image_width, 1)
    valid_rgb_values = torch.cat(3 * [alpha_images > 0.5], dim=3)
    rgb_images[~valid_rgb_values] = 0

    return torch.cat([rgb_images.float(), alpha_images], dim=3).flip([1])


def tone_mapper(image, gamma):
    """Applies gamma correction to the input image.

    Tone maps the input image batch in order to make scenes with a high dynamic
    range viewable. The gamma correction factor is computed separately per image,
    but is shared between all provided channels. The exact function computed is:

    image_out = A*image_in^gamma, where A is an image-wide constant computed so
    that the maximum image value is approximately 1. The correction is applied
    to all channels.

    Args:
        image: 4-D float32 tensor with shape [batch_size, image_height,
            image_width, channel_count]. The batch of images to tone map.
        gamma: 0-D float32 nonnegative tensor. Values of gamma below one compress
            relative contrast in the image, and values above one increase it. A
            value of 1 is equivalent to scaling the image to have a maximum value
            of 1.
    Returns:
        4-D float32 tensor with shape [batch_size, image_height, image_width,
        channel_count]. Contains the gamma-corrected images, clipped to the range
        [0, 1].
    """
    batch_size = image.shape[0]
    corrected_image = image ** gamma
    image_max, _ = corrected_image.view(batch_size, -1).max(dim=1)
    scaled_image = corrected_image / image_max.view(batch_size, 1, 1, 1)
    return scaled_image.clamp(0.0, 1.0)


def mesh_renderer(vertices,
                  triangles,
                  normals,
                  diffuse_colors,
                  camera_position,
                  camera_lookat,
                  camera_up,
                  light_positions,
                  light_intensities,
                  image_width,
                  image_height,
                  specular_colors=None,
                  shininess_coefficients=None,
                  ambient_color=None,
                  fov_y=40.0,
                  near_clip=0.01,
                  far_clip=10.0):
    """Renders an input scene using phong shading, and returns an output image.

    Args:
        vertices: 3-D float32 tensor with shape [batch_size, vertex_count, 3]. Each
            triplet is an xyz position in world space.
        triangles: 2-D int32 tensor with shape [triangle_count, 3]. Each triplet
            should contain vertex indices describing a triangle such that the
            triangle's normal points toward the viewer if the forward order of the
            triplet defines a clockwise winding of the vertices. Gradients with
            respect to this tensor are not available.
        normals: 3-D float32 tensor with shape [batch_size, vertex_count, 3]. Each
            triplet is the xyz vertex normal for its corresponding vertex. Each
            vector is assumed to be already normalized.
        diffuse_colors: 3-D float32 tensor with shape [batch_size,
            vertex_count, 3]. The RGB diffuse reflection in the range [0,1] for
            each vertex.
        camera_position: 2-D tensor with shape [batch_size, 3] or 1-D tensor with
            shape [3] specifying the XYZ world space camera position.
        camera_lookat: 2-D tensor with shape [batch_size, 3] or 1-D tensor with
            shape [3] containing an XYZ point along the center of the camera's gaze.
        camera_up: 2-D tensor with shape [batch_size, 3] or 1-D tensor with shape
            [3] containing the up direction for the camera. The camera will have no
            tilt with respect to this direction.
        light_positions: a 3-D tensor with shape [batch_size, light_count, 3]. The
            XYZ position of each light in the scene. In the same coordinate space as
            pixel_positions.
        light_intensities: a 3-D tensor with shape [batch_size, light_count, 3]. The
            RGB intensity values for each light. Intensities may be above one.
        image_width: int specifying desired output image width in pixels.
        image_height: int specifying desired output image height in pixels.
        specular_colors: 3-D float32 tensor with shape [batch_size,
            vertex_count, 3]. The RGB specular reflection in the range [0, 1] for
            each vertex.  If supplied, specular reflections will be computed, and
            both specular_colors and shininess_coefficients are expected.
        shininess_coefficients: a 0D-2D float32 tensor with maximum shape
            [batch_size, vertex_count]. The phong shininess coefficient of each
            vertex. A float gives a constant shininess coefficient
            across all batches and images. A 1D tensor must have shape [batch_size],
            and a single shininess coefficient per image is used.
        ambient_color: a 2D tensor with shape [batch_size, 3]. The RGB ambient
            color, which is added to each pixel in the scene. If None, it is
            assumed to be black.
        fov_y: float, or 1D tensor with shape [batch_size] specifying
            desired output image y field of view in degrees.
        near_clip: float, or 1D tensor with shape [batch_size] specifying
            near clipping plane distance.
        far_clip: float, or 1D tensor with shape [batch_size] specifying
            far clipping plane distance.

    Returns:
        renders: A 4-D float32 tensor of shape [batch_size, image_height, image_width, 4]
            containing the lit RGBA color values for each image at each pixel. RGB
            colors are the intensity values before tonemapping and can be in the range
            [0, infinity]. Clipping to the range [0,1] is likely
            reasonable for both viewing and training most scenes. More complex scenes
            with multiple lights should tone map color values for display only. One
            simple tonemapping approach is to rescale color values as x/(1+x); gamma
            compression is another common techinque. Alpha values are zero for
            background pixels and near one for mesh pixels.

        normalize_space_vertices: A 3-D float32 tensor of shape [batch_size, vertex_count, 2]
            containing the vertices that have been normalized in the image space. These normalized
            vertices can be used to generate landmarks.
    Raises:
        ValueError: An invalid argument to the method is detected.
    """

    if len(vertices.shape) != 3:
        raise ValueError('Vertices must have shape [batch_size, vertex_count, 3].')
    batch_size = vertices.shape[0]
    if len(triangles.shape) != 2:
        raise ValueError('Vertices must have shape [triangle_count, 3].')
    if len(normals.shape) != 3:
        raise ValueError('Normals must have shape [batch_size, vertex_count, 3].')
    if len(light_positions.shape) != 3:
        raise ValueError(
            'Light_positions must have shape [batch_size, light_count, 3].')
    if len(light_intensities.shape) != 3:
        raise ValueError(
            'Light_intensities must have shape [batch_size, light_count, 3].')
    if len(diffuse_colors.shape) != 3:
        raise ValueError(
            'vertex_diffuse_colors must have shape [batch_size, vertex_count, 3].')
    if (ambient_color is not None and
            list(ambient_color.shape) != [batch_size, 3]):
        raise ValueError('Ambient_color must have shape [batch_size, 3].')
    if list(camera_position.shape) == [3]:
        camera_position = camera_position.view(1, -1).expand([batch_size, -1])
    elif list(camera_position.shape) != [batch_size, 3]:
        raise ValueError('Camera_position must have shape [batch_size, 3]')
    if list(camera_lookat.shape) == [3]:
        camera_lookat = camera_lookat.view(1, -1).expand([batch_size, -1])
    elif list(camera_lookat.shape) != [batch_size, 3]:
        raise ValueError('Camera_lookat must have shape [batch_size, 3]')
    if list(camera_up.shape) == [3]:
        camera_up = camera_up.view(1, -1).expand([batch_size, -1])
    elif list(camera_up.shape) != [batch_size, 3]:
        raise ValueError('Camera_up must have shape [batch_size, 3]')
    if isinstance(fov_y, float):
        fov_y = torch.FloatTensor(batch_size * [fov_y]).to(vertices.device)
    elif list(fov_y.shape) != [batch_size]:
        raise ValueError('Fov_y must be a float or a 1D tensor with'
                         'shape [batch_size]')
    if isinstance(near_clip, float):
        near_clip = torch.FloatTensor(batch_size * [near_clip]).to(vertices.device)
    elif list(near_clip.shape) != [batch_size]:
        raise ValueError('Near_clip must be a float or a 1D tensor'
                         'with shape [batch_size]')
    if isinstance(far_clip, float):
        far_clip = torch.FloatTensor(batch_size * [far_clip]).to(vertices.device)
    elif list(far_clip.shape) != [batch_size]:
        raise ValueError('Far_clip must be a float, or a 1D tensor'
                         'with shape [batch_size]')
    if specular_colors is not None and shininess_coefficients is None:
        raise ValueError(
            'Specular colors were supplied without shininess coefficients.')
    if shininess_coefficients is not None and specular_colors is None:
        raise ValueError(
            'Shininess coefficients were supplied without specular colors.')
    if specular_colors is not None:
        # Since a 0-D float32 tensor is accepted, also accept a float.
        if isinstance(shininess_coefficients, float):
            shininess_coefficients = torch.FloatTensor([shininess_coefficients]).to(vertices.device)
        if len(specular_colors.shape) != 3:
            raise ValueError('The specular colors must have shape [batch_size, '
                             'vertex_count, 3].')
        if len(shininess_coefficients.shape) > 2:
            raise ValueError('The shininess coefficients must have shape at most'
                             '[batch_size, vertex_count].')
        # If we don't have per-vertex coefficients, we can just reshape the
        # input shininess to broadcast later, rather than interpolating an
        # additional vertex attribute:
        if len(shininess_coefficients.shape) < 2:
            vertex_attributes = torch.cat(
                [normals, vertices, diffuse_colors, specular_colors], dim=2)
        else:
            vertex_attributes = torch.cat(
                [
                    normals, vertices, diffuse_colors, specular_colors,
                    shininess_coefficients.unsqueeze(dim=2)
                ],
                dim=2)
    else:
        vertex_attributes = torch.cat([normals, vertices, diffuse_colors], dim=2)

    camera_matrices = look_at(camera_position, camera_lookat, camera_up)

    perspective_transforms = perspective(image_width / image_height, fov_y, near_clip, far_clip)

    clip_space_transforms = torch.matmul(perspective_transforms, camera_matrices)

    clip_space_vertices = transform_homogeneous(clip_space_transforms, vertices)
    pixel_attributes = rasterize_clip_space(clip_space_vertices, vertex_attributes, triangles,
                                            image_width, image_height, [-1] * vertex_attributes.shape[2])

    # Extract the interpolated vertex attributes from the pixel buffer and
    # supply them to the shader:
    pixel_normals = nn.functional.normalize(pixel_attributes[:, :, :, 0:3], p=2, dim=3)
    pixel_positions = pixel_attributes[:, :, :, 3:6]
    diffuse_colors = pixel_attributes[:, :, :, 6:9]

    if specular_colors is not None:
        specular_colors = pixel_attributes[:, :, :, 9:12]
        # Retrieve the interpolated shininess coefficients if necessary, or just
        # reshape our input for broadcasting:
        if len(shininess_coefficients.shape) == 2:
            shininess_coefficients = pixel_attributes[:, :, :, 12]
        else:
            shininess_coefficients = shininess_coefficients.view(-1, 1, 1)

    pixel_mask = (diffuse_colors >= 0).any(dim=3).float()

    renders = phong_shader(
        normals=pixel_normals,
        alphas=pixel_mask,
        pixel_positions=pixel_positions,
        light_positions=light_positions,
        light_intensities=light_intensities,
        diffuse_colors=diffuse_colors,
        camera_position=camera_position if specular_colors is not None else None,
        specular_colors=specular_colors,
        shininess_coefficients=shininess_coefficients,
        ambient_color=ambient_color)

    normalize_space_vertices = normalize_homogeneous(clip_space_vertices, image_width, image_height)
    return renders, normalize_space_vertices
