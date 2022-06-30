#!/usr/bin/env bash
sudo apt-get -qq install \
    gcc \
    libblas-dev \
    cmake

cd pyscf/pyscf/lib
curl -L https://github.com/fishjojo/pyscf-deps/blob/master/pyscf-2.0.1-ad-deps.tar.gz?raw=true | tar xzf -
mkdir build; cd build
cmake -DBUILD_LIBXC=OFF -DBUILD_XCFUN=OFF -DBUILD_LIBCINT=OFF ..
#cmake ..
make -j4
