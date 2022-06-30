#!/usr/bin/env bash
# MKL
wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
rm GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
sudo echo "deb https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt-get update
sudo apt-get install -y intel-oneapi-mkl
source /opt/intel/oneapi/setvars.sh
echo $MKLROOT
printenv >> $GITHUB_ENV

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
