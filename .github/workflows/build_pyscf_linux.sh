#!/usr/bin/env bash
sudo apt-get -qq install \
    gcc \
    libblas-dev \
    cmake

cd pyscf/pyscf/lib
#curl http://www.sunqm.net/pyscf/files/bin/pyscf-2.0a-deps.tar.gz | tar xzf -
mkdir build; cd build
#cmake -DBUILD_LIBXC=OFF -DBUILD_XCFUN=OFF -DBUILD_LIBCINT=OFF ..
cmake ..
make -j4
