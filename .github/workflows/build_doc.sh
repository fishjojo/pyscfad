#!/usr/bin/env bash
pip install geometric
pip install pyscf-properties
pip install pyscfad

cd doc

pip install -r requirements.txt

./scripts/gendoc.sh

python make.py html

touch build/html/.nojekyll
