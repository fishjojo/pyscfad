#!/usr/bin/env bash

export PYTHONPATH=$(pwd):$(pwd)/pyscfadlib:$PYTHONPATH
export OMP_NUM_THREADS=1

#pytest ./pyscfad --cov-report xml --cov=. --verbosity=1 --durations=10
pytest ./pyscfad/dft/test/test_uks.py
