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

# Modifications: this file integrates the test cases of 
# mesh_renderer_test.py and rasterize_triangles_test.py in original project
# Copyright 2020 Netease Fuxi AI LAB
# SPDX-License-Identifier: Apache-2.0

from torch.autograd.gradcheck import _as_tuple, _differentiable_outputs, warnings, get_analytical_jacobian, get_numerical_jacobian
import os
import unittest
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pt_mesh_renderer.camera_utils as camera_utils
import pt_mesh_renderer.kernels.rasterize_triangles as rasterize_triangles_kernels
from pt_mesh_renderer.RasterizeTriangles import rasterize_triangles
import pt_mesh_renderer.mesh_renderer as mesh_renderer
import pt_mesh_renderer.rasterizer as rasterizer
try:
    import pt_mesh_renderer.kernels.rasterize_triangles_cuda as rasterize_triangles_kernels_cuda
    is_cuda_valid = True
except Exception:
    is_cuda_valid = False
    print("Cannot import cuda rasterizer, renderer is running in CPU mode.")

test_data_directory = './test_data/'


def check_jacobians_are_nearly_equal(theoretical,
                                     numerical,
                                     outlier_relative_error_threshold,
                                     max_outlier_fraction,
                                     include_jacobians_in_error_message=True):
    """Compares two Jacobian matrices, allowing for some fraction of outliers.

    Args:
        theoretical: 2D numpy array containing a Jacobian matrix with entries
            computed via gradient functions. The layout should be as in the output
            of gradient_checker.
        numerical: 2D numpy array of the same shape as theoretical containing a
            Jacobian matrix with entries computed via finite difference
            approximations. The layout should be as in the output
            of gradient_checker.
        outlier_relative_error_threshold: float prescribing the maximum relative
            error (from the finite difference approximation) is tolerated before
            and entry is considered an outlier.
        max_outlier_fraction: float defining the maximum fraction of entries in
            theoretical that may be outliers before the check returns False.
        include_jacobians_in_error_message: bool defining whether the jacobian
            matrices should be included in the return message should the test fail.

    Returns:
        A tuple where the first entry is a boolean describing whether
        max_outlier_fraction was exceeded, and where the second entry is a string
        containing an error message if one is relevant.
    """
    outlier_gradients = np.abs(
        numerical - theoretical) / numerical > outlier_relative_error_threshold
    outlier_fraction = np.count_nonzero(outlier_gradients) / np.prod(
        numerical.shape[:2])
    jacobians_match = outlier_fraction <= max_outlier_fraction

    message = (
        ' %f of theoretical gradients are relative outliers, but the maximum'
        ' allowable fraction is %f ' % (outlier_fraction, max_outlier_fraction))
    if include_jacobians_in_error_message:
        # the gradient_checker convention is the typical Jacobian transposed:
        message += ('\nNumerical Jacobian:\n%s\nTheoretical Jacobian:\n%s' %
                    (repr(numerical.T), repr(theoretical.T)))
    return jacobians_match, message


# revise "torch.autograd.gradcheck" as "simple_gradcheck" for fitting original tf code
def simple_gradcheck(test_instance, func, inputs, eps,
                     outlier_relative_error_threshold,
                     max_outlier_fraction, nondet_tol=0.0):
    tupled_inputs = _as_tuple(inputs)

    func_out = func(*tupled_inputs)
    output = _differentiable_outputs(func_out)

    for i, o in enumerate(output):
        if not o.requires_grad:
            continue

        def fn(input):
            return _as_tuple(func(*input))[i]

        analytical, reentrant, correct_grad_sizes = get_analytical_jacobian(tupled_inputs, o, nondet_tol=nondet_tol)
        numerical = get_numerical_jacobian(fn, tupled_inputs, eps=eps)

        for j, (a, n) in enumerate(zip(analytical, numerical)):
            if a.numel() != 0 or n.numel() != 0:
                jacobians_match, message = (
                    check_jacobians_are_nearly_equal(
                        a, n,
                        outlier_relative_error_threshold,
                        max_outlier_fraction))
                test_instance.assertTrue(jacobians_match, message)


def expect_image_file_and_render_are_near(test_instance,
                                          baseline_name,
                                          result_image,
                                          max_outlier_fraction=0.001,
                                          pixel_error_threshold=0.01):
    """Compares the output of mesh_renderer with an image on disk.

    The comparison is soft: the images are considered identical if at most
    max_outlier_fraction of the pixels differ by more than a relative error of
    pixel_error_threshold of the full color value. Note that before comparison,
    mesh renderer values are clipped to the range [0,1].

    Uses _images_are_near for the actual comparison.

    Args:
        test_instance: a python unit test instance.
        baseline_name: path to the reference image in the test data dictionary.
        result_image: the result image, as a numpy array.
        max_outlier_fraction: the maximum fraction of outlier pixels allowed.
        pixel_error_threshold: pixel values are considered to differ if their
        difference exceeds this amount. Range is 0.0 - 1.0.
    """
    baseline_path = os.path.join(test_data_directory, baseline_name)
    baseline_image = plt.imread(baseline_path)

    test_instance.assertEqual(baseline_image.shape, result_image.shape,
                              'Image shapes %s and %s do not match.' %
                              (baseline_image.shape, result_image.shape))

    result_image = np.clip(result_image, 0., 1.).copy(order='C')
    baseline_image = baseline_image.astype(float)

    outlier_channels = (np.abs(baseline_image - result_image) >
                        pixel_error_threshold)

    outlier_pixels = np.any(outlier_channels, axis=2)
    outlier_count = np.count_nonzero(outlier_pixels)
    outlier_fraction = outlier_count / np.prod(baseline_image.shape[:2])
    images_match = outlier_fraction <= max_outlier_fraction

    outputs_dir = "./"
    base_prefix = os.path.splitext(os.path.basename(baseline_path))[0]
    result_output_path = os.path.join(outputs_dir, base_prefix + "_result.png")

    message = ('{} does not match. ({} of pixels are outliers, {} is allowed.). '
               'Result image written to {}'.format(
                   baseline_path, outlier_fraction, max_outlier_fraction, result_output_path))

    if not images_match:
        plt.subplot(131)
        plt.imshow(result_image)
        plt.subplot(132)
        plt.imshow(baseline_image)
        plt.subplot(133)
        plt.imshow(np.abs(baseline_image - result_image)[:, :, 0:3])
        plt.show()
        plt.imsave(result_output_path, result_image)

    test_instance.assertTrue(images_match, msg=message)


def highlight_print(string):
    length = len(string)
    print("")
    print("*" * (length+4))
    print("*", string, "*")
    print("*" * (length+4))


class RenderTest(unittest.TestCase):
    skip = False
    @classmethod
    def setUpClass(cls):
        highlight_print("RenderTest may take several minutes, please be patient...")

        # Set up a basic cube centered at the origin, with vertex normals pointing
        # outwards along the line from the origin to the cube vertices:
        cls.cube_vertices = torch.FloatTensor(
            [[-1, -1, 1], [-1, -1, -1], [-1, 1, -1], [-1, 1, 1], [1, -1, 1],
             [1, -1, -1], [1, 1, -1], [1, 1, 1]])

        cls.cube_normals = nn.functional.normalize(cls.cube_vertices, p=2, dim=1)
        cls.cube_triangles = torch.LongTensor(
            [[0, 1, 2], [2, 3, 0], [3, 2, 6], [6, 7, 3], [7, 6, 5], [5, 4, 7],
             [4, 5, 1], [1, 0, 4], [5, 6, 2], [2, 1, 5], [7, 4, 0], [0, 3, 7]])

    @classmethod
    def tearDownClass(cls):
        highlight_print("RenderTest done!")

    # run on multi-device
    def runOnMultiDevice(self, func):
        func("cpu")
        if is_cuda_valid:
            func("cuda:0")

    @unittest.skipIf(skip, "skip")
    def test_renderSimpleCube(self):
        """Renders a simple cube to test the full forward pass.

        Verifies the functionality of both the custom kernel and the python wrapper.
        """
        model_transforms = camera_utils.euler_matrices(
            torch.FloatTensor([[-20.0, 0.0, 60.0], [45.0, 60.0, 0.0]]))[:, :3, :3]

        vertices_world_space = torch.stack([self.cube_vertices] * 2).matmul(
            model_transforms.transpose(1, 2))

        normals_world_space = torch.stack([self.cube_normals] * 2).matmul(
            model_transforms.transpose(1, 2))

        # camera position:
        eye = torch.FloatTensor(2 * [[0.0, 0.0, 6.0]])
        center = torch.FloatTensor(2 * [[0.0, 0.0, 0.0]])
        world_up = torch.FloatTensor(2 * [[0.0, 1.0, 0.0]])
        image_width = 640
        image_height = 480
        light_positions = torch.FloatTensor([[[0.0, 0.0, 6.0]], [[0.0, 0.0, 6.0]]])
        light_intensities = torch.ones([2, 1, 3], dtype=torch.float32)
        vertex_diffuse_colors = torch.ones_like(vertices_world_space)

        # execute rendering
        def runOnDevice(device):
            renders, _ = mesh_renderer.mesh_renderer(
                vertices_world_space.to(device), self.cube_triangles.to(device),
                normals_world_space.to(device), vertex_diffuse_colors.to(device),
                eye.to(device), center.to(device), world_up.to(device), light_positions.to(device),
                light_intensities.to(device), image_width, image_height)

            for image_id in range(renders.shape[0]):
                target_image_name = 'Gray_Cube_{}.png'.format(image_id)
                expect_image_file_and_render_are_near(
                    self, target_image_name, renders[image_id, :, :, :].cpu().numpy())

        self.runOnMultiDevice(runOnDevice)

    @unittest.skipIf(skip, "skip")
    def test_complexShading(self):
        """Tests specular highlights, colors, and multiple lights per image."""
        # rotate the cube for the test:
        model_transforms = camera_utils.euler_matrices(
            torch.FloatTensor([[-20.0, 0.0, 60.0], [45.0, 60.0, 0.0]]))[:, :3, :3]

        vertices_world_space = torch.stack([self.cube_vertices] * 2).matmul(
            model_transforms.transpose(1, 2))

        normals_world_space = torch.stack([self.cube_normals] * 2).matmul(
            model_transforms.transpose(1, 2))

        # camera position:
        eye = torch.FloatTensor([[0.0, 0.0, 6.0], [0., 0.2, 18.0]])
        center = torch.FloatTensor([[0.0, 0.0, 0.0], [0.1, -0.1, 0.1]])
        world_up = torch.FloatTensor([[0.0, 1.0, 0.0], [0.1, 1.0, 0.15]])
        fov_y = torch.FloatTensor([40., 13.3])
        near_clip = 0.1
        far_clip = 25.
        image_width = 640
        image_height = 480
        light_positions = torch.FloatTensor([[[0.0, 0.0, 6.0], [1.0, 2.0, 6.0]],
                                             [[0.0, -2.0, 4.0], [1.0, 3.0, 4.0]]])
        light_intensities = torch.FloatTensor([[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                                               [[2.0, 0.0, 1.0], [0.0, 2.0, 1.0]]])
        # colors
        vertex_diffuse_colors = torch.FloatTensor(2*[[[1.0, 0.0, 0.0],
                                                      [0.0, 1.0, 0.0],
                                                      [0.0, 0.0, 1.0],
                                                      [1.0, 1.0, 1.0],
                                                      [1.0, 1.0, 0.0],
                                                      [1.0, 0.0, 1.0],
                                                      [0.0, 1.0, 1.0],
                                                      [0.5, 0.5, 0.5]]])
        vertex_specular_colors = torch.FloatTensor(2*[[[0.0, 1.0, 0.0],
                                                       [0.0, 0.0, 1.0],
                                                       [1.0, 1.0, 1.0],
                                                       [1.0, 1.0, 0.0],
                                                       [1.0, 0.0, 1.0],
                                                       [0.0, 1.0, 1.0],
                                                       [0.5, 0.5, 0.5],
                                                       [1.0, 0.0, 0.0]]])
        shininess_coefficients = 6.0 * torch.ones([2, 8], dtype=torch.float32)
        ambient_color = torch.FloatTensor([[0., 0., 0.], [0.1, 0.1, 0.2]])

        # execute rendering
        def runOnDevice(device):
            renders, _ = mesh_renderer.mesh_renderer(
                vertices_world_space.to(device), self.cube_triangles.to(device),
                normals_world_space.to(device), vertex_diffuse_colors.to(device),
                eye.to(device), center.to(device), world_up.to(device), light_positions.to(device),
                light_intensities.to(device), image_width, image_height,
                vertex_specular_colors.to(device), shininess_coefficients.to(device),
                ambient_color.to(device), fov_y.to(device), near_clip, far_clip)

            tonemapped_renders = torch.cat(
                [
                    mesh_renderer.tone_mapper(renders[:, :, :, 0:3], 0.7),
                    renders[:, :, :, 3:4]
                ],
                dim=3)

            # Check that shininess coefficient broadcasting works by also rendering
            # with a scalar shininess coefficient, and ensuring the result is identical:
            broadcasted_renders, _ = mesh_renderer.mesh_renderer(
                vertices_world_space.to(device), self.cube_triangles.to(device),
                normals_world_space.to(device), vertex_diffuse_colors.to(device),
                eye.to(device), center.to(device), world_up.to(device), light_positions.to(device),
                light_intensities.to(device), image_width, image_height,
                vertex_specular_colors.to(device), 6.0,
                ambient_color.to(device), fov_y.to(device), near_clip, far_clip)

            tonemapped_broadcasted_renders = torch.cat(
                [
                    mesh_renderer.tone_mapper(broadcasted_renders[:, :, :, 0:3], 0.7),
                    broadcasted_renders[:, :, :, 3:4]
                ],
                dim=3)

            for image_id in range(renders.shape[0]):
                target_image_name = 'Colored_Cube_{}.png'.format(image_id)
                expect_image_file_and_render_are_near(
                    self, target_image_name,
                    tonemapped_renders[image_id, :, :, :].cpu().numpy())
                expect_image_file_and_render_are_near(
                    self, target_image_name,
                    tonemapped_broadcasted_renders[image_id, :, :, :].cpu().numpy())

        self.runOnMultiDevice(runOnDevice)

    @unittest.skipIf(skip, "skip")
    def test_fullRenderGradientComputation(self):
        """Verifies the Jacobian matrix for the entire renderer.

        This ensures correct gradients are propagated backwards through the entire
        process, not just through the rasterization kernel. Uses the simple cube
        forward pass.
        """
        image_height = 21
        image_width = 28

        # rotate the cube for the test:
        model_transforms = camera_utils.euler_matrices(
            torch.FloatTensor([[-20.0, 0.0, 60.0], [45.0, 60.0, 0.0]]))[:, :3, :3]

        # camera position:
        eye = torch.FloatTensor([0.0, 0.0, 6.0])
        center = torch.FloatTensor([0.0, 0.0, 0.0])
        world_up = torch.FloatTensor([0.0, 1.0, 0.0])

        # Scene has a single light from the viewer's eye.
        light_positions = torch.cat([eye.view(1, 1, 3)]*2, dim=0)
        light_intensities = torch.ones([2, 1, 3], dtype=torch.float32)

        vertex_diffuse_colors = torch.ones([2, 8, 3], dtype=torch.float32)

        def renderer(cube_vertices):
            device = cube_vertices.device
            vertices_world_space = torch.stack([cube_vertices] * 2).matmul(
                model_transforms.to(device).transpose(1, 2))
            # cube_normals = nn.functional.normalize(cube_vertices, p=2, dim=1)
            normals_world_space = torch.stack([self.cube_normals.to(device)] * 2).matmul(
                model_transforms.to(device).transpose(1, 2))
            rendered, _ = mesh_renderer.mesh_renderer(
                vertices_world_space.to(device), self.cube_triangles.to(device),
                normals_world_space.to(device), vertex_diffuse_colors.to(device),
                eye.to(device), center.to(device), world_up.to(device), light_positions.to(device),
                light_intensities.to(device), image_width, image_height)
            return rendered

        def runOnDevice(device):
            cube_vertices = self.cube_vertices.clone().to(device).float().requires_grad_(True)
            simple_gradcheck(self, renderer, cube_vertices, 1e-3, 0.01, 0.01)

        self.runOnMultiDevice(runOnDevice)

    @unittest.skipIf(skip, "skip")
    def test_thatCubeRotates(self):
        """Optimize a simple cube's rotation using pixel loss.

        The rotation is represented as static-basis euler angles. This test checks
        that the computed gradients are useful.
        """
        image_height = 480
        image_width = 640
        initial_euler_angles = [[0.0, 0.0, 0.0]]

        # camera position:
        eye = torch.FloatTensor([[0.0, 0.0, 6.0]])
        center = torch.FloatTensor([[0.0, 0.0, 0.0]])
        world_up = torch.FloatTensor([[0.0, 1.0, 0.0]])

        vertex_diffuse_colors = torch.ones([1, 8, 3], dtype=torch.float32)
        light_positions = eye.view(1, 1, 3)
        light_intensities = torch.ones([1, 1, 3], dtype=torch.float32)

        def render(euler_angles):
            device = euler_angles.device
            model_rotation = camera_utils.euler_matrices(euler_angles)[:, :3, :3]
            vertices_world_space = self.cube_vertices.to(device).matmul(model_rotation.transpose(1, 2)).view(1, 8, 3)
            normals_world_space = self.cube_normals.to(device).matmul(model_rotation.transpose(1, 2)).view(1, 8, 3)

            rendered, _ = mesh_renderer.mesh_renderer(
                vertices_world_space, self.cube_triangles.to(device), normals_world_space,
                vertex_diffuse_colors.to(device), eye.to(device), center.to(device), world_up.to(device),
                light_positions.to(device), light_intensities.to(device),
                image_width, image_height)
            return rendered.view(image_height, image_width, 4)

        # Pick the desired cube rotation for the test:
        test_model_rotation = torch.FloatTensor([[-20.0, 0.0, 60.0]])

        with torch.no_grad():
            desired_rendered = render(test_model_rotation)

        def runOnDevice(device):
            euler_angles = torch.nn.Parameter(torch.FloatTensor(initial_euler_angles).to(device))
            optimizer = torch.optim.SGD([euler_angles], 0.7, momentum=0.1)
            for i in range(35):
                optimizer.zero_grad()
                rendered = render(euler_angles)
                loss = torch.nn.functional.l1_loss(rendered, desired_rendered.to(device))
                loss.backward()
                optimizer.step()
                if False:
                    print("step: {} loss: {:.6f}".format(i, loss.item()))

            with torch.no_grad():
                final_rendered = render(euler_angles).cpu()

            target_image_name = 'Gray_Cube_0.png'
            expect_image_file_and_render_are_near(
                self, target_image_name, desired_rendered.numpy())
            expect_image_file_and_render_are_near(
                self, target_image_name, final_rendered.numpy(),
                max_outlier_fraction=0.01, pixel_error_threshold=0.04)

        self.runOnMultiDevice(runOnDevice)


class RasterizeTest(unittest.TestCase):
    skip = False
    @classmethod
    def setUpClass(cls):
        highlight_print("RasterizeTest may take several minutes, please be patient...")

        # Set up a basic cube centered at the origin, with vertex normals pointing
        # outwards along the line from the origin to the cube vertices:
        cls.cube_vertex_positions = torch.FloatTensor(
            [[-1, -1, 1], [-1, -1, -1], [-1, 1, -1], [-1, 1, 1], [1, -1, 1],
             [1, -1, -1], [1, 1, -1], [1, 1, 1]])

        cls.cube_triangles = torch.LongTensor(
            [[0, 1, 2], [2, 3, 0], [3, 2, 6], [6, 7, 3], [7, 6, 5], [5, 4, 7],
             [4, 5, 1], [1, 0, 4], [5, 6, 2], [2, 1, 5], [7, 4, 0], [0, 3, 7]])

        cls.image_width = 640
        cls.image_height = 480

        cls.perspective = camera_utils.perspective(
            cls.image_width / cls.image_height,
            torch.FloatTensor([40.0]),
            torch.FloatTensor([0.01]),
            torch.FloatTensor([10.0]))

    @classmethod
    def tearDownClass(cls):
        highlight_print("RasterizeTest done!")

    def run_triangleTest(self, w_vector, target_image_name):
        """Directly renders a rasterized triangle's barycentric coordinates.

        Tests only the kernel (rasterize_triangles_module).

        Args:
        w_vector: 3 element vector of w components to scale triangle vertices.
        target_image_name: image file name to compare result against.
        """
        clip_init = np.array(
            [[-0.5, -0.5, 0.8, 1.0], [0.0, 0.5, 0.3, 1.0], [0.5, -0.5, 0.3, 1.0]],
            dtype=np.float32)
        clip_init = clip_init * np.reshape(
            np.array(w_vector, dtype=np.float32), [3, 1])

        clip_coordinates = torch.FloatTensor(clip_init)
        triangles = torch.LongTensor([[0, 1, 2]])

        rendered_coordinates, _, _ = rasterize_triangles(
            clip_coordinates, triangles, self.image_width, self.image_height)
        rendered_coordinates = torch.cat(
            [rendered_coordinates, torch.ones([self.image_height, self.image_width, 1])], dim=2)

        image = rendered_coordinates.numpy()
        expect_image_file_and_render_are_near(self, target_image_name, image)

    @unittest.skipIf(skip, "skip")
    def test_rendersSimpleTriangle(self):
        self.run_triangleTest((1.0, 1.0, 1.0), 'Simple_Triangle.png')

    @unittest.skipIf(skip, "skip")
    def test_rendersPerspectiveCorrectTriangle(self):
        self.run_triangleTest((0.2, 0.5, 2.0), 'Perspective_Corrected_Triangle.png')

    @unittest.skipIf(skip, "skip")
    def testRendersTwoCubesInBatch(self):
        """Renders a simple cube in two viewpoints to test the python wrapper."""

        vertex_rgb = (self.cube_vertex_positions * 0.5 + 0.5)
        vertex_rgba = torch.cat([vertex_rgb, torch.ones([8, 1])], dim=1)

        center = torch.FloatTensor([[0.0, 0.0, 0.0]])
        world_up = torch.FloatTensor([[0.0, 1.0, 0.0]])
        look_at_1 = camera_utils.look_at(torch.FloatTensor([[2.0, 3.0, 6.0]]),
                                         center, world_up)
        look_at_2 = camera_utils.look_at(torch.FloatTensor([[-3.0, 1.0, 6.0]]),
                                         center, world_up)
        projection_1 = self.perspective.matmul(look_at_1)
        projection_2 = self.perspective.matmul(look_at_2)
        projection = torch.cat([projection_1, projection_2], dim=0)
        background_value = [0.0, 0.0, 0.0, 0.0]

        rendered = rasterizer.rasterize(
            torch.stack([self.cube_vertex_positions, self.cube_vertex_positions]),
            torch.stack([vertex_rgba, vertex_rgba]), self.cube_triangles, projection,
            self.image_width, self.image_height, background_value)

        for i in (0, 1):
            image = rendered[i, :, :, :].numpy()
            baseline_image_name = 'Unlit_Cube_{}.png'.format(i)
            expect_image_file_and_render_are_near(self, baseline_image_name, image)

    @unittest.skipIf(skip, "skip")
    def test_simpleTriangleGradientComputation(self):
        """Verifies the Jacobian matrix for a single pixel.

        The pixel is in the center of a triangle facing the camera. This makes it
        easy to check which entries of the Jacobian might not make sense without
        worrying about corner cases.
        """
        test_pixel_x = 325
        test_pixel_y = 245

        triangles = torch.LongTensor([[0, 1, 2]])

        def triangle_rasterizer(clip_coordinates):
            barycentric_coordinates, _, _ = rasterize_triangles(
                clip_coordinates, triangles, self.image_width, self.image_height)
            pixels_to_compare = barycentric_coordinates[
                test_pixel_y:test_pixel_y + 1, test_pixel_x:test_pixel_x + 1, :]
            return pixels_to_compare

        clip_coordinates = torch.FloatTensor(
            [[-0.5, -0.5, 0.8, 1.0], [0.0, 0.5, 0.3, 1.0], [0.5, -0.5, 0.3, 1.0]]).requires_grad_(True)

        # simple_gradcheck(self, triangle_rasterizer, clip_coordinates, 4e-2, 0.01, 0)
        # print("Warning: the gradient may not be precise, so the threshold is temporarily set to 0.1 from 0.")
        simple_gradcheck(self, triangle_rasterizer, clip_coordinates, 4e-2, 0.01, 0.1)

    @unittest.skipIf(skip, "skip")
    def test_internalRenderGradientComputation(self):
        """Isolates and verifies the Jacobian matrix for the custom kernel."""
        image_height = 21
        image_width = 28

        def triangle_rasterizer(clip_coordinates):
            barycentric_coordinates, _, _ = rasterize_triangles(
                clip_coordinates, self.cube_triangles, image_width, image_height)
            return barycentric_coordinates

        # Precomputed transformation of the simple cube to normalized device
        # coordinates, in order to isolate the rasterization gradient.
        # pyformat: disable
        clip_coordinates = torch.FloatTensor(
            [[-0.43889722, -0.53184521, 0.85293502, 1.0],
             [-0.37635487, 0.22206162, 0.90555805, 1.0],
             [-0.22849123, 0.76811147, 0.80993629, 1.0],
             [-0.2805393, -0.14092168, 0.71602166, 1.0],
             [0.18631913, -0.62634289, 0.88603103, 1.0],
             [0.16183566, 0.08129397, 0.93020856, 1.0],
             [0.44147962, 0.53497446, 0.85076219, 1.0],
             [0.53008741, -0.31276882, 0.77620775, 1.0]]).requires_grad_(True)

        simple_gradcheck(self, triangle_rasterizer, clip_coordinates, 4e-2, 0.01, 0.01)


class RasterizeKernelTest(unittest.TestCase):
    skip = False
    @classmethod
    def setUpClass(cls):
        highlight_print("RasterizeKernelTest may take several minutes, please be patient...")

        cls.image_width = 640
        cls.image_height = 480

    @classmethod
    def tearDownClass(cls):
        highlight_print("RasterizeKernelTest done!")

    def RGB2RGBA(self, src):
        alpha = torch.ones([self.image_height, self.image_width, 1])
        return torch.cat([src, alpha], dim=2)

    @unittest.skipIf(skip, "skip")
    def test_canRasterizeTriangle(self):
        vertices = torch.FloatTensor([-0.5, -0.5, 0.8, 1.0,  0.0, 0.5,
                                      0.3,  1.0,  0.5, -0.5, 0.3, 1.0]).view(-1, 4)
        triangles = torch.LongTensor([0, 1, 2]).view(-1, 3)

        barycentrics, _, _ = rasterize_triangles_kernels.forward_rasterize_triangles(
            vertices, triangles, self.image_width, self.image_height)

        expect_image_file_and_render_are_near(
            self, "Simple_Triangle.png", self.RGB2RGBA(barycentrics).numpy())

    @unittest.skipIf(skip, "skip")
    def test_canRasterizeExternalTriangle(self):
        vertices = torch.FloatTensor([-0.5, -0.5, 0.0, 1.0,  0.0, -0.5,
                                      0.0,  -1.0, 0.5, -0.5, 0.0, 1.0]).view(-1, 4)
        triangles = torch.LongTensor([0, 1, 2]).view(-1, 3)

        barycentrics, _, _ = rasterize_triangles_kernels.forward_rasterize_triangles(
            vertices, triangles, self.image_width, self.image_height)

        expect_image_file_and_render_are_near(
            self, "External_Triangle.png", barycentrics.numpy())

    @unittest.skipIf(skip, "skip")
    def test_canRasterizeCameraInsideBox(self):
        vertices = torch.FloatTensor([
            -1.0, -1.0, 0.0, 2.0, 1.0, -1.0, 0.0, 2.0, 1.0, 1.0, 0.0,
            2.0, -1.0, 1.0, 0.0, 2.0, -1.0, -1.0, 0.0, -2.0, 1.0, -1.0,
            0.0, -2.0, 1.0, 1.0, 0.0, -2.0, -1.0, 1.0, 0.0, -2.0
        ]).view(-1, 4)
        triangles = torch.LongTensor([
            0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7,
            2, 3, 7, 2, 7, 6, 1, 0, 4, 1, 4, 5,
            0, 3, 7, 0, 7, 4, 1, 2, 6, 1, 6, 5
        ]).view(-1, 3)

        barycentrics, _, _ = rasterize_triangles_kernels.forward_rasterize_triangles(
            vertices, triangles, self.image_width, self.image_height)

        expect_image_file_and_render_are_near(
            self, "Inside_Box.png", barycentrics.numpy())

    @unittest.skipIf(skip, "skip")
    def test_canRasterizeTetrahedron(self):
        vertices = torch.FloatTensor([
            -0.5, -0.5, 0.8, 1.0,  0.0, 0.5,
            0.3,  1.0,  0.5, -0.5, 0.3, 1.0,
            0.0,  0.0,  0.0, 1.0
        ]).view(-1, 4)
        triangles = torch.LongTensor([
            0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3
        ]).view(-1, 3)

        barycentrics, _, _ = rasterize_triangles_kernels.forward_rasterize_triangles(
            vertices, triangles, self.image_width, self.image_height)

        expect_image_file_and_render_are_near(
            self, "Simple_Tetrahedron.png", barycentrics.numpy())

    @unittest.skipIf(skip, "skip")
    def test_canRasterizeCube(self):
        """  Vertex values were obtained by dumping the clip-space vertex values from
        the test_renderSimpleCube test in RenderTest.
        """

        vertices = torch.FloatTensor([
            -2.60648608, -3.22707772,  6.85085106, 6.85714293,
            -1.30324292, -0.992946863, 8.56856918, 8.5714283,
            -1.30324292, 3.97178817,   7.70971,    7.71428585,
            -2.60648608, 1.73765731,   5.991992,   6,
            1.30324292,  -3.97178817,  6.27827835, 6.28571415,
            2.60648608,  -1.73765731,  7.99599648, 8,
            2.60648608,  3.22707772,   7.13713741, 7.14285707,
            1.30324292,  0.992946863,  5.41941929, 5.4285717
        ]).view(-1, 4)
        triangles = torch.LongTensor([
            0, 1, 2, 2, 3, 0, 3, 2, 6, 6, 7, 3,
            7, 6, 5, 5, 4, 7, 4, 5, 1, 1, 0, 4,
            5, 6, 2, 2, 1, 5, 7, 4, 0, 0, 3, 7
        ]).view(-1, 3)

        barycentrics, _, _ = rasterize_triangles_kernels.forward_rasterize_triangles(
            vertices, triangles, self.image_width, self.image_height)

        expect_image_file_and_render_are_near(
            self, "Barycentrics_Cube.png", barycentrics.numpy())

    def expectBarycentricSumIsNear(self, barycentric_val, expected):
        """Expects that the sum of barycentric weights at a pixel is close to a given value."""
        kEpsilon = 1e-6
        message = ('Barycentric sum ({}) is not near to {} ({} is allowed.). '.format(
            barycentric_val, expected, kEpsilon))
        is_near = np.abs(barycentric_val - expected) < kEpsilon
        self.assertTrue(is_near, msg=message)

    def expectIsCovered(self, barycentric_val):
        """Expects that a pixel is covered by verifying that its barycentric coordinates sum to one."""
        self.expectBarycentricSumIsNear(barycentric_val, 1.0)

    def expectIsNotCovered(self, barycentric_val):
        """Expects that a pixel is not covered by verifying that its barycentric coordinates sum to zero."""
        self.expectBarycentricSumIsNear(barycentric_val, 0.0)

    @unittest.skipIf(skip, "skip")
    def test_worksWhenPixelIsOnTriangleEdge(self):
        image_width = 641
        x_pixel = int(image_width / 2)
        x_ndc = 0.0
        yPixel = 5

        vertices = torch.FloatTensor([
            x_ndc, -1.0, 0.5, 1.0,  x_ndc, 1.0,
            0.5,   1.0,  0.5, -1.0, 0.5,   1.0
        ]).view(-1, 4)

        triangles = torch.LongTensor([0, 1, 2]).view(-1, 3)
        barycentrics, _, _ = rasterize_triangles_kernels.forward_rasterize_triangles(
            vertices, triangles, self.image_width, self.image_height)
        self.expectIsCovered(barycentrics[yPixel, x_pixel, :].sum())

        triangles = torch.LongTensor([2, 1, 0]).view(-1, 3)
        barycentrics, _, _ = rasterize_triangles_kernels.forward_rasterize_triangles(
            vertices, triangles, self.image_width, self.image_height)
        self.expectIsCovered(barycentrics[yPixel, x_pixel, :].sum())

    @unittest.skipIf(skip, "skip")
    def test_coversEdgePixelsOfImage(self):
        """Verifies that the pixels along image edges are correct covered."""

        vertices = torch.FloatTensor([
            -1.0, -1.0, 0.0, 1.0, 1.0, -1.0,
            0.0,  1.0,  1.0, 1.0, 0.0, 1.0,
            -1.0, 1.0,  0.0, 1.0
        ]).view(-1, 4)
        triangles = torch.LongTensor([0, 1, 2, 0, 2, 3]).view(-1, 3)

        barycentrics, _, _ = rasterize_triangles_kernels.forward_rasterize_triangles(
            vertices, triangles, self.image_width, self.image_height)

        self.expectIsCovered(barycentrics[0, 0, :].sum())
        self.expectIsCovered(barycentrics[self.image_height - 1, 0, :].sum())
        self.expectIsCovered(barycentrics[self.image_height - 1, self.image_width - 1, :].sum())
        self.expectIsCovered(barycentrics[0, self.image_width - 1, :].sum())

    @unittest.skipIf(skip, "skip")
    def test_pixelOnDegenerateTriangleIsNotInside(self):
        """Verifies that the pixels along image edges are correct covered."""
        image_width = 1
        image_height = 1

        vertices = torch.FloatTensor([
            -1.0, -1.0, 0.0, 1.0, 1.0, 1.0,
            0.0,  1.0,  0.0, 0.0, 0.0, 1.0
        ]).view(-1, 4)
        triangles = torch.LongTensor([0, 1, 2]).view(-1, 3)

        barycentrics, _, _ = rasterize_triangles_kernels.forward_rasterize_triangles(
            vertices, triangles, image_width, image_height)

        self.expectIsNotCovered(barycentrics[0, 0, :].sum())


class RasterizeKernelCudaTest(unittest.TestCase):
    skip = False and is_cuda_valid
    @classmethod
    def setUpClass(cls):
        highlight_print("RasterizeKernelCudaTest may take several minutes, please be patient...")

        cls.image_width = 640
        cls.image_height = 480

    @classmethod
    def tearDownClass(cls):
        highlight_print("RasterizeKernelCudaTest done!")

    def RGB2RGBA(self, src):
        alpha = torch.ones([self.image_height, self.image_width, 1])
        return torch.cat([src, alpha], dim=2)

    @unittest.skipIf(skip, "skip")
    def test_canRasterizeTriangleCuda(self):
        vertices = torch.FloatTensor([-0.5, -0.5, 0.8, 1.0,  0.0, 0.5,
                                      0.3,  1.0,  0.5, -0.5, 0.3, 1.0]).view(-1, 4)
        triangles = torch.LongTensor([0, 1, 2]).view(-1, 3)

        barycentrics, _, _ = rasterize_triangles_kernels_cuda.forward_rasterize_triangles_cuda(
            vertices.cuda(), triangles.cuda(), self.image_width, self.image_height)

        expect_image_file_and_render_are_near(
            self, "Simple_Triangle.png", self.RGB2RGBA(barycentrics.cpu()).numpy())

    @unittest.skipIf(skip, "skip")
    def test_canRasterizeExternalTriangleCuda(self):
        vertices = torch.FloatTensor([-0.5, -0.5, 0.0, 1.0,  0.0, -0.5,
                                      0.0,  -1.0, 0.5, -0.5, 0.0, 1.0]).view(-1, 4)
        triangles = torch.LongTensor([0, 1, 2]).view(-1, 3)

        barycentrics, _, _ = rasterize_triangles_kernels_cuda.forward_rasterize_triangles_cuda(
            vertices.cuda(), triangles.cuda(), self.image_width, self.image_height)

        expect_image_file_and_render_are_near(
            self, "External_Triangle.png", barycentrics.cpu().numpy())

    @unittest.skipIf(skip, "skip")
    def test_canRasterizeCameraInsideBoxCuda(self):
        vertices = torch.FloatTensor([
            -1.0, -1.0, 0.0, 2.0, 1.0, -1.0, 0.0, 2.0, 1.0, 1.0, 0.0,
            2.0, -1.0, 1.0, 0.0, 2.0, -1.0, -1.0, 0.0, -2.0, 1.0, -1.0,
            0.0, -2.0, 1.0, 1.0, 0.0, -2.0, -1.0, 1.0, 0.0, -2.0
        ]).view(-1, 4)
        triangles = torch.LongTensor([
            0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7,
            2, 3, 7, 2, 7, 6, 1, 0, 4, 1, 4, 5,
            0, 3, 7, 0, 7, 4, 1, 2, 6, 1, 6, 5
        ]).view(-1, 3)

        barycentrics, _, _ = rasterize_triangles_kernels_cuda.forward_rasterize_triangles_cuda(
            vertices.cuda(), triangles.cuda(), self.image_width, self.image_height)

        expect_image_file_and_render_are_near(
            self, "Inside_Box.png", barycentrics.cpu().numpy())

    @unittest.skipIf(skip, "skip")
    def test_canRasterizeTetrahedronCuda(self):
        vertices = torch.FloatTensor([
            -0.5, -0.5, 0.8, 1.0,  0.0, 0.5,
            0.3,  1.0,  0.5, -0.5, 0.3, 1.0,
            0.0,  0.0,  0.0, 1.0
        ]).view(-1, 4)
        triangles = torch.LongTensor([
            0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3
        ]).view(-1, 3)

        barycentrics, _, _ = rasterize_triangles_kernels_cuda.forward_rasterize_triangles_cuda(
            vertices.cuda(), triangles.cuda(), self.image_width, self.image_height)

        expect_image_file_and_render_are_near(
            self, "Simple_Tetrahedron.png", barycentrics.cpu().numpy())

    @unittest.skipIf(skip, "skip")
    def test_canRasterizeCubeCuda(self):
        """  Vertex values were obtained by dumping the clip-space vertex values from
        the test_renderSimpleCube test in RenderTest.
        """

        vertices = torch.FloatTensor([
            -2.60648608, -3.22707772,  6.85085106, 6.85714293,
            -1.30324292, -0.992946863, 8.56856918, 8.5714283,
            -1.30324292, 3.97178817,   7.70971,    7.71428585,
            -2.60648608, 1.73765731,   5.991992,   6,
            1.30324292,  -3.97178817,  6.27827835, 6.28571415,
            2.60648608,  -1.73765731,  7.99599648, 8,
            2.60648608,  3.22707772,   7.13713741, 7.14285707,
            1.30324292,  0.992946863,  5.41941929, 5.4285717
        ]).view(-1, 4)
        triangles = torch.LongTensor([
            0, 1, 2, 2, 3, 0, 3, 2, 6, 6, 7, 3,
            7, 6, 5, 5, 4, 7, 4, 5, 1, 1, 0, 4,
            5, 6, 2, 2, 1, 5, 7, 4, 0, 0, 3, 7
        ]).view(-1, 3)

        barycentrics, _, _ = rasterize_triangles_kernels_cuda.forward_rasterize_triangles_cuda(
            vertices.cuda(), triangles.cuda(), self.image_width, self.image_height)

        expect_image_file_and_render_are_near(
            self, "Barycentrics_Cube.png", barycentrics.cpu().numpy())

    def expectBarycentricSumIsNear(self, barycentric_val, expected):
        """Expects that the sum of barycentric weights at a pixel is close to a given value."""
        kEpsilon = 1e-6
        message = ('Barycentric sum ({}) is not near to {} ({} is allowed.). '.format(
            barycentric_val, expected, kEpsilon))
        is_near = np.abs(barycentric_val - expected) < kEpsilon
        self.assertTrue(is_near, msg=message)

    def expectIsCovered(self, barycentric_val):
        """Expects that a pixel is covered by verifying that its barycentric coordinates sum to one."""
        self.expectBarycentricSumIsNear(barycentric_val, 1.0)

    def expectIsNotCovered(self, barycentric_val):
        """Expects that a pixel is not covered by verifying that its barycentric coordinates sum to zero."""
        self.expectBarycentricSumIsNear(barycentric_val, 0.0)

    @unittest.skipIf(skip, "skip")
    def test_worksWhenPixelIsOnTriangleEdgeCuda(self):
        image_width = 641
        x_pixel = int(image_width / 2)
        x_ndc = 0.0
        yPixel = 5

        vertices = torch.FloatTensor([
            x_ndc, -1.0, 0.5, 1.0,  x_ndc, 1.0,
            0.5,   1.0,  0.5, -1.0, 0.5,   1.0
        ]).view(-1, 4)

        triangles = torch.LongTensor([0, 1, 2]).view(-1, 3)
        barycentrics, _, _ = rasterize_triangles_kernels_cuda.forward_rasterize_triangles_cuda(
            vertices.cuda(), triangles.cuda(), self.image_width, self.image_height)
        barycentrics = barycentrics.cpu()
        self.expectIsCovered(barycentrics[yPixel, x_pixel, :].sum())

        triangles = torch.LongTensor([2, 1, 0]).view(-1, 3)
        barycentrics, _, _ = rasterize_triangles_kernels_cuda.forward_rasterize_triangles_cuda(
            vertices.cuda(), triangles.cuda(), self.image_width, self.image_height)
        barycentrics = barycentrics.cpu()
        self.expectIsCovered(barycentrics[yPixel, x_pixel, :].sum())

    @unittest.skipIf(skip, "skip")
    def test_coversEdgePixelsOfImageCuda(self):
        """Verifies that the pixels along image edges are correct covered."""

        vertices = torch.FloatTensor([
            -1.0, -1.0, 0.0, 1.0, 1.0, -1.0,
            0.0,  1.0,  1.0, 1.0, 0.0, 1.0,
            -1.0, 1.0,  0.0, 1.0
        ]).view(-1, 4)
        triangles = torch.LongTensor([0, 1, 2, 0, 2, 3]).view(-1, 3)

        barycentrics, _, _ = rasterize_triangles_kernels_cuda.forward_rasterize_triangles_cuda(
            vertices.cuda(), triangles.cuda(), self.image_width, self.image_height)
        barycentrics = barycentrics.cpu()

        self.expectIsCovered(barycentrics[0, 0, :].sum())
        self.expectIsCovered(barycentrics[self.image_height - 1, 0, :].sum())
        self.expectIsCovered(barycentrics[self.image_height - 1, self.image_width - 1, :].sum())
        self.expectIsCovered(barycentrics[0, self.image_width - 1, :].sum())

    @unittest.skipIf(skip, "skip")
    def test_pixelOnDegenerateTriangleIsNotInsideCuda(self):
        """Verifies that the pixels along image edges are correct covered."""
        image_width = 1
        image_height = 1

        vertices = torch.FloatTensor([
            -1.0, -1.0, 0.0, 1.0, 1.0, 1.0,
            0.0,  1.0,  0.0, 0.0, 0.0, 1.0
        ]).view(-1, 4)
        triangles = torch.LongTensor([0, 1, 2]).view(-1, 3)

        barycentrics, _, _ = rasterize_triangles_kernels_cuda.forward_rasterize_triangles_cuda(
            vertices.cuda(), triangles.cuda(), image_width, image_height)
        barycentrics = barycentrics.cpu()

        self.expectIsNotCovered(barycentrics[0, 0, :].sum())


if __name__ == "__main__":
    unittest.main()
