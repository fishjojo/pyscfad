#!/usr/bin/env bash
python -m pip install --upgrade pip
python -m pip cache purge
pip install wheel
pip install pytest
pip install pytest-cov
pip install 'numpy<2.4'
pip install 'jax<0.9'
pip install 'pyscf<2.12.0'
