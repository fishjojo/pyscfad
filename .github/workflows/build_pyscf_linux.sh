#!/usr/bin/env bash
sudo apt-get -qq install \
    gcc \
    gfortran \
    libgfortran3 \
    libblas-dev \
    cmake \
    curl

cd pyscf/pyscf/lib
#curl http://www.sunqm.net/pyscf/files/bin/pyscf-1.7.5-deps.tar.gz | tar xzf -
mkdir build; cd build
#cmake -DBUILD_LIBXC=OFF -DBUILD_XCFUN=OFF ..
cmake ..
make -j4
#cd ../../../..
