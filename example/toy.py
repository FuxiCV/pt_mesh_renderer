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

# Modifications: a toy code for cpu/cuda verification
# Copyright 2020 Netease Fuxi AI LAB
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import numpy as np
import pt_mesh_renderer.camera_utils as camera_utils
import pt_mesh_renderer.mesh_renderer as mesh_renderer
import matplotlib.pyplot as plt


class Cube(nn.Module):
    """
    Render a cube based on pt_mesh_renderer and input rotation parameter.
    """
    def __init__(self):
        super(Cube, self).__init__()
        self.image_height = 480
        self.image_width = 640

        # cube info:
        cube_vertices = torch.FloatTensor(
            [[-1, -1, 1], [-1, -1, -1], [-1, 1, -1], [-1, 1, 1], [1, -1, 1],
             [1, -1, -1], [1, 1, -1], [1, 1, 1]])

        cube_normals = nn.functional.normalize(cube_vertices, p=2, dim=1)
        cube_triangles = torch.LongTensor(
            [[0, 1, 2], [2, 3, 0], [3, 2, 6], [6, 7, 3], [7, 6, 5], [5, 4, 7],
             [4, 5, 1], [1, 0, 4], [5, 6, 2], [2, 1, 5], [7, 4, 0], [0, 3, 7]])

        self.cube_vertices = nn.Parameter(cube_vertices, requires_grad=False)
        self.cube_normals = nn.Parameter(cube_normals, requires_grad=False)
        self.cube_triangles = nn.Parameter(cube_triangles, requires_grad=False)

        # camera position:
        eye = torch.FloatTensor([[0.0, 0.0, 6.0]])
        center = torch.FloatTensor([[0.0, 0.0, 0.0]])
        world_up = torch.FloatTensor([[0.0, 1.0, 0.0]])

        vertex_diffuse_colors = torch.ones([1, 8, 3], dtype=torch.float32)
        light_positions = eye.view(1, 1, 3)
        light_intensities = torch.ones([1, 1, 3], dtype=torch.float32)

        self.eye = nn.Parameter(eye, requires_grad=False)
        self.center = nn.Parameter(center, requires_grad=False)
        self.world_up = nn.Parameter(world_up, requires_grad=False)
        self.vertex_diffuse_colors = nn.Parameter(vertex_diffuse_colors, requires_grad=False)
        self.light_positions = nn.Parameter(light_positions, requires_grad=False)
        self.light_intensities = nn.Parameter(light_intensities, requires_grad=False)

    def forward(self, euler_angles):
        model_rotation = camera_utils.euler_matrices(euler_angles)[:, :3, :3]
        vertices_world_space = self.cube_vertices.matmul(model_rotation.transpose(1, 2)).view(1, 8, 3)
        normals_world_space = self.cube_normals.matmul(model_rotation.transpose(1, 2)).view(1, 8, 3)

        rendered, geometry = mesh_renderer.mesh_renderer(
            vertices_world_space, self.cube_triangles, normals_world_space,
            self.vertex_diffuse_colors, self.eye, self.center, self.world_up,
            self.light_positions, self.light_intensities,
            self.image_width, self.image_height)
        return rendered.view(self.image_height, self.image_width, 4), geometry


if __name__ == "__main__":
    device = "cpu" #or "cuda:0"
    cube = Cube().to(device)
    
    # Pick the desired cube rotation for the test:
    with torch.no_grad():
        target_rotation = torch.FloatTensor([[-20.0, 0.0, 60.0]]).to(device) / 180 * np.pi
        target_rendered, target_geometry = cube(target_rotation)

    # initialization
    initial_euler_angles = [[0.0, 0.0, 0.0]]
    euler_angles = torch.nn.Parameter(torch.FloatTensor(initial_euler_angles).to(device))
    optimizer = torch.optim.Adam([euler_angles], 1e-1)

    # optimizing
    for i in range(35):
        optimizer.zero_grad()
        rendered, geometry = cube(euler_angles)
        loss_img = torch.nn.functional.l1_loss(rendered, target_rendered)
        loss_geo = 1e-3 * torch.nn.functional.l1_loss(geometry, target_geometry)
        loss = loss_img + loss_geo
        loss.backward()
        optimizer.step()

        print("step: {} loss: {:.6f} loss_img: {:.6f} loss_geo: {:.6f}".format(i, loss.item(), loss_img, loss_geo))
        if False:
            plt.ion()
            plt.figure(1)
            plt.clf()
            plt.subplot(121)
            plt.imshow(rendered.detach().cpu().numpy())
            geo = geometry.detach().cpu().numpy()
            plt.scatter(geo[:, :, 0], geo[:, :, 1])
            plt.subplot(122)
            plt.imshow(target_rendered.detach().cpu().numpy())
            dgeo = target_geometry.detach().cpu().numpy()
            plt.scatter(dgeo[:, :, 0], dgeo[:, :, 1])
            plt.pause(0.1)
    plt.ioff()
    plt.show()
