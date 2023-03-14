#!/usr/bin/env bash
export OMP_NUM_THREADS=1
export PYTHONPATH=$(pwd):$(pwd)/pyscf:$PYTHONPATH
# preload MKL
#if [[ -n "${MKLROOT}" ]]; then
#  export LD_PRELOAD=$MKLROOT/lib/intel64/libmkl_avx2.so:$MKLROOT/lib/intel64/libmkl_sequential.so:$MKLROOT/lib/intel64/libmkl_core.so
#fi
echo "pyscfad = True" >> $HOME/.pyscf_conf.py
echo "pyscf_numpy_backend = 'jax'" >> $HOME/.pyscf_conf.py
echo "pyscf_scipy_linalg_backend = 'pyscfad'" >> $HOME/.pyscf_conf.py
echo "pyscf_scipy_backend = 'jax'" >> $HOME/.pyscf_conf.py
echo "pyscfad_scf_implicit_diff = True" >> $HOME/.pyscf_conf.py
echo "pyscfad_ccsd_implicit_diff = True" >> $HOME/.pyscf_conf.py

cd pyscfad
pytest --cov-report xml --cov=. --verbosity=1 --durations=10
