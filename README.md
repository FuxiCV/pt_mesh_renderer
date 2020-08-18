# Mesh Renderer (PyTorh version)

This is a PyTorch implementation of tf_mesh_renderer [[paper](https://arxiv.org/abs/1806.06098)][[project](https://github.com/google/tf_mesh_renderer)], and has been applied in Game Character Auto-Creation [[paper](https://arxiv.org/abs/2008.07154)].

Different from the original project, this PyTorch version supports `Windows OS` and `CUDA acceleration`.

## How to start

* install

```cmd
export TORCH_CUDA_ARCH_LIST=7.0 # may be required by GTX2080Ti
git clone https://github.com/FuxiCV/pt_mesh_renderer
cd pt_mesh_renderer
python setup.py install
```

* test

```cmd
cd pt_mesh_renderer/
python pt_mesh_renderer_test.py
```

* demo

```cmd
cd example/
python toy.py
```

* uninstall

```cmd
pip uninstall pt_mesh_renderer
```

* env

    This code has been verified on:
    
    * `Ubuntu 18.04` with `Python3.7`, `CUDA10.1` and `PyTorch 1.4.0 & 1.5.0`

    * `windows 10 x64` with `VS2019`, `Python3.7`, `CUDA10.1` and `PyTorch 1.4.0 & 1.5.0` (Please refer to the following notes)

## Citation

If you use this renderer in your research, please cite [original Tensorflow version](http://openaccess.thecvf.com/content_cvpr_2018/html/Genova_Unsupervised_Training_for_CVPR_2018_paper.html "Tensorflow version") and [this PyTorch version](https://arxiv.org/abs/2008.07154 "PyTorch version")

```
@InProceedings{Genova_2018_CVPR,
  author = {Genova, Kyle and Cole, Forrester and Maschinot, Aaron and Sarna, Aaron and Vlasic, Daniel and Freeman, William T.},
  title = {Unsupervised Training for 3D Morphable Model Regression},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2018}
}
```

```
@misc{shi2020neutral,
    title={Neutral Face Game Character Auto-Creation via PokerFace-GAN},
    author={Tianyang Shi and Zhengxia Zou and Xinhui Song and Zheng Song and Changjian Gu and Changjie Fan and Yi Yuan},
    year={2020},
    eprint={2008.07154},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```


## Note: Install from local clone on Windows

* Reference: https://github.com/facebookresearch/pytorch3d => [INSTALL.md](https://github.com/facebookresearch/pytorch3d/blob/master/INSTALL.md)
* The following tricks are also valid on pytorch 1.5 but some file paths may be changed

If you are using pre-compiled pytorch 1.4 and torchvision 0.5, you should make the following changes to the pytorch source code to successfully compile with Visual Studio 2019 (MSVC 19.16.27034) and CUDA 10.1.

Change python/Lib/site-packages/torch/include/csrc/jit/script/module.h

L466, 476, 493, 506, 536
```
-static constexpr *
+static const *
```
Change python/Lib/site-packages/torch/include/csrc/jit/argument_spec.h

L190
```
-static constexpr size_t DEPTH_LIMIT = 128;
+static const size_t DEPTH_LIMIT = 128;
```

Change python/Lib/site-packages/torch/include/pybind11/cast.h

L1449
```
-explicit operator type&() { return *(this->value); }
+explicit operator type& () { return *((type*)(this->value)); }
```

After patching, you can go to "x64 Native Tools Command Prompt for VS 2019" to compile and install
```
cd pt_mesh_renderer
python3 setup.py install
```
After installing, verify whether all unit tests have passed
```
python3 pt_mesh_renderer_test.py
```
