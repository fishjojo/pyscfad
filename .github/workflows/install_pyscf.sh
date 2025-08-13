#!/usr/bin/env bash
python -m pip install --upgrade pip
python -m pip cache purge
pip install wheel
pip install pytest
pip install pytest-cov
pip install numpy
pip install scipy
pip install h5py
pip install 'jaxlib>=0.7.0'
pip install 'jax>=0.7.0'
pip install 'pyscf>=2.10.0'
