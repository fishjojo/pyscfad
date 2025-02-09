#!/usr/bin/env bash

export PYTHONPATH=$(pwd):$(pwd)/pyscfadlib:$PYTHONPATH

pytest ./pyscfad --cov-report xml --cov=. --verbosity=1 --durations=10
