#!/usr/bin/env bash

export PYTHONPATH=$(pwd):$(pwd)/pyscfadlib:$PYTHONPATH

export OMP_NUM_THREADS=2
pytest ./pyscfad --cov-report xml --cov=. --verbosity=1 --durations=10
