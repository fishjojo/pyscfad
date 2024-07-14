#!/usr/bin/env bash

cd pyscfadlib/pyscfadlib
mkdir build
cmake -B build
cmake --build build -j2
rm -rf build
