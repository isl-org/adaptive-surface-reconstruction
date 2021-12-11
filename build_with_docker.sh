#!/bin/bash
set -euo pipefail

function printhelp {
    __usage="
Usage: $(basename $0) [OPTIONS]

Options:
    --wheel <python version>  Build a python wheel with the specified version
    --appimage                Build the AppImage
    --help, -h                Print this help  
"
    echo "$__usage"
    exit 0
}

PYTHON_VERSION=3.9
BUILD_PYTHON=OFF
BUILD_APPIMAGE=OFF

if [ $# -eq 0 ]; then
    printhelp
fi

parse_iter=0
while [ $parse_iter -lt 100 ] ; do
    parse_iter=$((parse_iter+1))
    case "$1" in
        --wheel) PYTHON_VERSION="$2"; BUILD_PYTHON=ON ; shift 2 ;;
        --appimage) BUILD_APPIMAGE=ON ; shift ;;
        --help | -h) printhelp ; shift ;;
        *) break ;;
    esac
done

REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

docker build -t adaptive-surface-reconstruction .

docker run \
    --rm \
    --workdir /workspace \
    -e PYTHON_VERSION \
    -e BUILD_PYTHON \
    -e BUILD_APPIMAGE \
    -v "$REPO_ROOT":/workspace
    adaptive-surface-reconstruction
