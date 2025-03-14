#!/bin/bash

set -ex

ulimit -n 1024
#ln -s libquadmath.so.0 /usr/lib64/libquadmath.so
yum install -y yum-utils
yum install -y epel-release
yum-config-manager --enable epel
yum install -y openblas-devel
yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
#yum install -y cuda-toolkit
yum install -y \
    cuda-nvcc-12-8-12.8.93-1 \
    cuda-cudart-devel-12-8-12.8.90-1 \
    libcublas-devel-12-8-12.8.4.1-1 \
    libcusolver-devel-12-8-11.7.3.90-1 \
    libcusparse-devel-12-8-12.5.8.93-1 \
    libnvjitlink-devel-12-8-12.8.93-1
