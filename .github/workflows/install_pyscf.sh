#!/usr/bin/env bash
python -m pip install --upgrade pip
python -m pip cache purge
pip install wheel
pip install numpy
pip install 'scipy<1.12'
pip install h5py
pip install jaxlib
pip install jax
pip install pytest
pip install pytest-cov

pip install 'pyscf==2.3'
pip install 'pyscfadlib==0.1.4'
