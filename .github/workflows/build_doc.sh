#!/usr/bin/env bash
sudo apt-get install -y pandoc
pip install pandoc

pip install geometric
pip install pyscf-properties

export PYTHONPATH=$(pwd):$(pwd)/pyscfadlib:$PYTHONPATH

cd doc
pip install -r requirements.txt

python make.py --python-path $(pwd)/..:$(pwd)/../pyscfadlib html

touch build/html/.nojekyll
