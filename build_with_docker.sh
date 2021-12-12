#!/bin/bash

function printhelp {
    usage="
Usage: $(basename $0) [OPTIONS]

Example: 
    ./$(basename $0) --wheel 3.9 --appimage

Options:
    --wheel <python version>  Build a python wheel with the specified version
    --appimage                Build the AppImage
    --help, -h                Print this help  
"
    echo "$usage"
    exit 0
}

PYTHON_VERSION=3.9
BUILD_PYTHON_MODULE=OFF
BUILD_APPIMAGE=OFF

if [ $# -eq 0 ]; then
    printhelp
fi

parse_iter=0
while [ $parse_iter -lt 100 ] ; do
    parse_iter=$((parse_iter+1))
    case "$1" in
        --wheel) PYTHON_VERSION="$2"; BUILD_PYTHON_MODULE=ON ; shift 2 ;;
        --appimage) BUILD_APPIMAGE=ON ; shift ;;
        --help | -h) printhelp ; shift ;;
        *) break ;;
    esac
done
set -euo pipefail

REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

docker build -t adaptive-surface-reconstruction .

docker run \
    --rm \
    --workdir /workspace \
    -e PYTHON_VERSION="$PYTHON_VERSION" \
    -e BUILD_PYTHON_MODULE="$BUILD_PYTHON_MODULE" \
    -e BUILD_APPIMAGE="$BUILD_APPIMAGE" \
    -e MAKE_FLAGS="-j" \
    -v "$REPO_ROOT":/workspace \
    adaptive-surface-reconstruction
