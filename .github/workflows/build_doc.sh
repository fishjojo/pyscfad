#!/usr/bin/env bash
sudo apt-get install -y pandoc
pip install pandoc

pip install geometric
pip install pyscf-properties

cd pyscfadlib
python setup.py install
cd ..

pip install -r requirements.txt
export PYTHONPATH=$(pwd):$PYTHONPATH

cd doc

pip install -r requirements.txt

python make.py html

touch build/html/.nojekyll
