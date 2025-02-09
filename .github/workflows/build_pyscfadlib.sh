#!/usr/bin/env bash

sudo apt-get -qq install liblapack-dev

pip install nanobind

cd pyscfadlib/pyscfadlib
mkdir build
cmake -B build
cmake --build build -j2
rm -rf build
