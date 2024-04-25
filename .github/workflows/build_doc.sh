#!/usr/bin/env bash
export PYTHONPATH=$(pwd):$PYTHONPATH

cd doc
pip install -r requirements.txt
python make.py html

cd ..
rm -r docs
mv doc/build/html docs
touch docs/.nojekyll
