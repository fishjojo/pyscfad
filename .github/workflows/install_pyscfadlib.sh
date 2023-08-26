#!/usr/bin/env bash
cd pyscfadlib
mkdir build
cd build
cmake ..
make
cd ..
rm -rf build
