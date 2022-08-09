PySCF with Auto-differentiation
===============================

[![Build Status](https://github.com/fishjojo/pyscfad/workflows/CI/badge.svg)](https://github.com/fishjojo/pyscfad/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/fishjojo/pyscfad/branch/main/graph/badge.svg?token=NLSWGI0PLE)](https://codecov.io/gh/fishjojo/pyscfad)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6960749.svg)](https://doi.org/10.5281/zenodo.6960749)

Installation
------------

* To install the latest release, use the following command:
```
# install pyscf
pip install 'pyscf @ git+https://github.com/fishjojo/pyscf.git@ad#egg=pyscf',
pip install 'pyscf-properties @ git+https://github.com/fishjojo/properties.git@ad',
# install pyscfad
pip install pyscfad
```

* To install the development version, use the following command instead:
```
pip install git+https://github.com/fishjojo/pyscfad.git
```

* Install dependencies manually
```
pip install numpy scipy h5py
pip install jax jaxlib jaxopt

# install pyscf
cd $HOME; git clone https://github.com/fishjojo/pyscf.git
cd pyscf; git checkout ad 
cd pyscf/lib; mkdir build 
cd build; cmake ..; make

export PYTHONPATH=$HOME/pyscf:$PYTHONPATH
```

Running examples
----------------

* Add the following lines to the PySCF configure file ($HOME/.pyscf\_conf.py)
```
pyscfad = True
pyscf_numpy_backend = 'jax'
pyscf_scipy_linalg_backend = 'pyscfad'
pyscf_scipy_backend = 'jax'
# The followings are optional
pyscfad_scf_implicit_diff = True
pyscfad_ccsd_implicit_diff = True
```
