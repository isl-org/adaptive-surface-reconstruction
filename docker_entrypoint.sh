#!/bin/bash
set -euo pipefail

echo "================================================================================"
echo " PYTHON_VERSION         $PYTHON_VERSION"
echo " BUILD_PYTHON_MODULE    $BUILD_PYTHON_MODULE"
echo " BUILD_APPIMAGE         $BUILD_APPIMAGE"
echo " MAKE_FLAGS             $MAKE_FLAGS"
echo "================================================================================"

conda create -y -n asr python=$PYTHON_VERSION
source activate asr
conda install -y cmake

python -m pip install torch==1.13.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install open3d==0.14.1 zstandard msgpack msgpack-numpy

pushd datasets
python create_t10k_msgpacks.py --attribution_file_only
popd
if [ -d build ]; then
    rm -rf build
fi
mkdir build
pushd build

if [ "$BUILD_APPIMAGE" = "ON" ]; then
    mkdir linuxdeploy
    pushd linuxdeploy
    wget https://github.com/linuxdeploy/linuxdeploy/releases/download/continuous/linuxdeploy-x86_64.AppImage
    chmod u+x linuxdeploy-x86_64.AppImage

    # FUSE does not work inside docker -> extract app image
    ./linuxdeploy-x86_64.AppImage --appimage-extract
    rm linuxdeploy-x86_64.AppImage 
    popd
fi
linuxdeploy="$(pwd)/linuxdeploy/squashfs-root/AppRun"

cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_PYTHON_MODULE=$BUILD_PYTHON_MODULE \
    -DBUILD_APPIMAGE=$BUILD_APPIMAGE \
    -Dlinuxdeploy_binary="$linuxdeploy" \
    ..

df -h
make $MAKE_FLAGS

mv python/python_package/dist/a*.whl appimage/asr*.AppImage . || true
popd
