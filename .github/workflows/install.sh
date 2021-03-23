#!/usr/bin/env bash
python -m pip install --upgrade pip
pip install wheel
pip install numpy scipy h5py 
pip install git+https://github.com/fishjojo/pyscf.git@ad 
pip install jaxlib jax 
pip install pytest
