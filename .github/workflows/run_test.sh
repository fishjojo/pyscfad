#!/usr/bin/env bash
export OMP_NUM_THREADS=1
export PYTHONPATH=$(pwd):$(pwd)/pyscf:$PYTHONPATH
echo "pyscfad = True" >> $HOME/.pyscf_conf.py
#echo "pyscf_numpy_backend = 'jax'" >> $HOME/.pyscf_conf.py
#echo "pyscf_scipy_linalg_backend = 'pyscfad'" >> $HOME/.pyscf_conf.py
#echo "pyscf_scipy_backend = 'jax'" >> $HOME/.pyscf_conf.py

pytest ./pyscfad --cov-report xml --cov=. --verbosity=1 --durations=10
