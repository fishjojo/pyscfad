#!/usr/bin/env bash
sudo apt-get install -y pandoc
pip install pandoc

pip install geometric
pip install pyscf-properties
pip install pyscfad

cd doc

pip install -r requirements.txt

python make.py html

touch build/html/.nojekyll
