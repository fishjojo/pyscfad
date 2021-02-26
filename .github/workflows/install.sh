#!/usr/bin/env bash
python -m pip install --upgrade pip
pip install numpy scipy h5py pyscf jaxlib jax flax pytest
export OMP_NUM_THREADS=1
export PYTHONPATH=$(pwd):$PYTHONPATH
