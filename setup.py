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

# Modifications: Pytorch implementation of tf_mesh_renderer
# Copyright 2020 Netease Fuxi AI LAB
# SPDX-License-Identifier: Apache-2.0

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import pt_mesh_renderer
import sys

cppExtension = CppExtension('pt_mesh_renderer.kernels.rasterize_triangles', [
    'pt_mesh_renderer/kernels/rasterize_triangles.cc',
    'pt_mesh_renderer/kernels/rasterize_triangles_impl.cc'
])

cudaExtension = CUDAExtension('pt_mesh_renderer.kernels.rasterize_triangles_cuda', [
    'pt_mesh_renderer/kernels/rasterize_triangles_cuda.cc',
    'pt_mesh_renderer/kernels/rasterize_triangles_cuda_impl.cu'
])

ext_modules = [cppExtension, cudaExtension]

setup(
    description='PyTorch implementation of tf_mesh_renderer (with cuda)',
    author='shitianyang (Netease Fuxi AI LAB)',
    author_email='shitianyang@corp.netease.com',
    url='The reference code comes from https://github.com/google/tf_mesh_renderer',
    license='Apache-2.0 License',
    version=pt_mesh_renderer.__version__,
    name='pt_mesh_renderer',
    test_suite='tests',
    packages=['pt_mesh_renderer'],
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension}
)
