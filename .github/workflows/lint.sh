#!/usr/bin/env bash

export PYTHONPATH=$(pwd):$(pwd)/pyscfadlib:$PYTHONPATH

pylint pyscfad
