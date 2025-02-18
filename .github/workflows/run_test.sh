#!/usr/bin/env bash

export PYTHONPATH=$(pwd):$(pwd)/pyscfadlib:$PYTHONPATH
export OMP_NUM_THREADS=1

#pytest ./pyscfad --cov-report xml --cov=. --verbosity=1 --durations=10

coverage erase

MODULES=("scipy" "gto" "scf" "dft" "cc" "fci" "gw" "mp" "tdscf" "lo" "pbc")

for mod in "${MODULES[@]}"; do
    pytest "./pyscfad/$mod" --cov=pyscfad --cov-report=xml --verbosity=1 --durations=5 --cov-append
done

#coverage report -m
coverage xml
