#!/bin/bash

set -ex

echo "$CIBW_ARCHS"
echo "$CIBW_ARCHS_LINUX"

if [ "$CIBW_ARCHS_LINUX" == "x86_64" ]; then
    # Install CUDA Toolkit
    yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
    yum install -y cuda-toolkit-12-5-12.5.1-1
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    nvcc --version
    export PYSCFADLIB_ENABLE_CUDA=True
fi
