#!/usr/bin/env bash
export OMP_NUM_THREADS=1
export PYTHONPATH=$(pwd):$PYTHONPATH

pytest ./pyscfad --cov-report xml --cov=. --verbosity=1 --durations=10
