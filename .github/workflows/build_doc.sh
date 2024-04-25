#!/usr/bin/env bash
export PYTHONPATH=$(pwd):$PYTHONPATH

cd doc
pip install -r requirements.txt
python make.py html
