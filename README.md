PySCF with Auto-differentiation
===============================

[![Build Status](https://github.com/fishjojo/pyscfad/workflows/CI/badge.svg)](https://github.com/fishjojo/pyscfad/actions?query=workflow%3ACI)

Installation
------------

* Install to the python site-packages folder
```
pip install git+https://github.com/fishjojo/pyscfad.git
```

* Install manually
```
pip install numpy scipy h5py
pip install jaxlib jax

# install pyscf
cd $HOME; git clone https://github.com/fishjojo/pyscf.git
cd pyscf; git checkout ad 
cd pyscf/lib; mkdir build 
cd build; cmake ..; make

# install pyscfad
cd $HOME; git clone https://github.com/fishjojo/pyscfad.git
cd pyscfad/pyscfad/lib; mkdir build
cd build; cmake ..; make

export PYTHONPATH=$HOME/pyscf:$HOME/pyscfad:$PYTHONPATH
```

Running examples
----------------

* Add the following line to the PySCF configure file ($HOME/.pyscf\_conf.py)
```
pyscfad = True
```
