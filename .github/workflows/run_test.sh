#!/usr/bin/env bash
export OMP_NUM_THREADS=1

pytest ./pyscfad --cov-report xml --cov=. --verbosity=1 --durations=10
