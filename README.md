# Adaptive Surface Reconstruction

This repository contains code for the ICCV 2021 paper

*B. Ummenhofer and V. Koltun. "Adaptive Surface Reconstruction with Multiscale Convolutional Kernels". ICCV 2021.*


The code implements our surface reconstruction, which can fuse large scale
point clouds to create surfaces with varying levels of details.

If you find this repository useful please cite our [paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Ummenhofer_Adaptive_Surface_Reconstruction_With_Multiscale_Convolutional_Kernels_ICCV_2021_paper.pdf).

```
@InProceedings{Ummenhofer_2021_ICCV,
    author    = {Ummenhofer, Benjamin and Koltun, Vladlen},
    title     = {Adaptive Surface Reconstruction With Multiscale Convolutional Kernels},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {5651-5660}
}
```


## Dependencies

## Packages for building the library
- Pytorch 1.8.2 (can be installed with `python -m pip install torch==1.8.2+cpu -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html`)
- On Ubuntu the following packages are required: patchelf, xorg-dev, libglu1-mesa-dev, python3-dev
  These can be installed with `apt install patchelf xorg-dev libglu1-mesa-dev python3-dev`

## Packages required for training the network
- Tensorflow 2.6.0
- Open3D 0.14 or later with ML module (https://github.com/isl-org/Open3D/)
- Tensorpack DataFlow (for reading data, ```pip install --upgrade git+https://github.com/tensorpack/dataflow.git```)
- python-prctl (needed by Tensorpack DataFlow; depends on libcap-dev, install with ```apt install libcap-dev``` )
- msgpack (```pip install msgpack``` )
- msgpack-numpy (```pip install msgpack-numpy```)
- python-zstandard (```pip install zstandard``` https://github.com/indygreg/python-zstandard)
- SciPy

The versions match the configuration that we have tested on a system with Ubuntu 18.04.
We recommend using the latest versions for all packages.


## Build instructions

The library, python bindings, and example binary can be build with

```bash
mkdir build
cd build
cmake ..
make
```

The python package can be installed globally with the target ```pip-install-package```.
```bash
# inside the build directory
make install-pip-package
```

A portable AppImage of the binary can be created with the target ```appimage```.
This requires the linuxdeploy tool from https://github.com/linuxdeploy/linuxdeploy/releases/tag/continuous
```bash
# inside the build directory
# creates appimage/asrtool-0.1.0-x86_64.AppImage inside the build directory
make appimage 
```


## Directory structure

The project consists of a python module, a cpp library, an example binary, and training code.
Note that the python module is required for the training code to work.
The following gives an overview of how the code is organized.
```
├─ appimage               # Scripts and resources for building an AppImage for the binary
├─ cmake                  # CMake files for finding dependencies
├─ cpp
    ├─ bin                # Code for the example binary
    ├─ lib                # Code and headers for the library
    ├─ pybind             # Code for the python binding
├─ models                 # Code for training the models
├─ python                 # Python code and scripts for the python module
├─ utils                  # Contains general utility scripts
    ├─ deeplearningutils  # General training utils for tf/torch
```

## License

Code and scripts are under the Apache-2.0 license.
