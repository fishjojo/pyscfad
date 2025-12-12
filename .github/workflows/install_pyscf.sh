#!/usr/bin/env bash
python -m pip install --upgrade pip
python -m pip cache purge
pip install wheel
pip install pytest
pip install pytest-cov
pip install 'jax==0.8.1'
pip install 'pyscf==2.11.0'
