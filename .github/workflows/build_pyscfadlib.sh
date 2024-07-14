#!/usr/bin/env bash

pip install nanobind

cd pyscfadlib/pyscfadlib
mkdir build
cmake -B build
cmake --build build -j2
rm -rf build
