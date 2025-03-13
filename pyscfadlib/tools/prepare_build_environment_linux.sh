#!/bin/bash

set -ex

#if [ "$CIBW_ARCHS_LINUX" == "x86_64" ]; then
# Install CUDA Toolkit
yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
yum install -y cuda-toolkit-12-12.4.1-1
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
nvcc --version
#fi
