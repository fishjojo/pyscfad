PySCF with Auto-differentiation
===============================

[![Build Status](https://github.com/fishjojo/pyscfad/workflows/CI/badge.svg)](https://github.com/fishjojo/pyscfad/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/fishjojo/pyscfad/branch/main/graph/badge.svg?token=NLSWGI0PLE)](https://codecov.io/gh/fishjojo/pyscfad)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6960749.svg)](https://doi.org/10.5281/zenodo.6960749)

Installation
------------

---
* To install the latest release, use the following commands:
```
# install pyscfad
pip install pyscfad

# install pyscfadlib, optional
pip install pyscfadlib
```

---
* To install the development version, use the following command instead:
```
pip install git+https://github.com/fishjojo/pyscfad.git
```


Running examples
----------------

* In order to perform AD calculations, 
the following lines need to be added to 
the PySCF configure file($HOME/.pyscf\_conf.py)
```
pyscfad = True
```
