#!/usr/bin/env bash

git clone https://github.com/pyscf/pyscf.git
cd pyscf
git checkout 'v2.3.0'

cp -rf pyscf/lib/vhf ../vhf
cp -rf pyscf/lib/gto ../gto
cp -rf pyscf/lib/np_helper ../np_helper
