PySCF with Auto-differentiation
===============================

[![Build Status](https://github.com/fishjojo/pyscfad/workflows/CI/badge.svg)](https://github.com/fishjojo/pyscfad/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/fishjojo/pyscfad/branch/main/graph/badge.svg?token=NLSWGI0PLE)](https://codecov.io/gh/fishjojo/pyscfad)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6960749.svg)](https://doi.org/10.5281/zenodo.6960749)

Installation
------------

---
* To install the latest release, run:
```
pip install pyscfad
```

---
* To install the development version, use the following command instead:
```
pip install git+https://github.com/fishjojo/pyscfad.git
```
This should also install all the dependencies.
Alternatively, the dependencies can be installed via a predefined conda environment:
```
conda env create -f environment.yml
conda activate pyscfad_env
```
OpenMP is not required, but is recommended:
```
# install OpenMP runtime used with clang
# On Linux:
sudo apt update
sudo apt install libomp-dev

# On OSX:
brew install libomp
```

---
* One can also run PySCFAD inside a docker container:
```
docker pull fishjojo/pyscfad:latest
docker run -rm -t -i fishjojo/pyscfad:latest /bin/bash
```

Running examples
----------------

* In order to perform AD calculations, 
the following lines need to be added to 
the PySCF configure file ($HOME/.pyscf\_conf.py)
```
pyscfad = True
pyscf_numpy_backend = 'jax'
pyscf_scipy_linalg_backend = 'pyscfad'
pyscf_scipy_backend = 'jax'
# The followings turn on implicit differentiations
# for SCF and CC amplitude solvers
pyscfad_scf_implicit_diff = True
pyscfad_ccsd_implicit_diff = True
```

Citing PySCFAD
--------------
The following paper should be cited in publications utilizing the PySCFAD program package:

[Differentiable quantum chemistry with PySCF for molecules and materials at the mean-field level and beyond](https://doi.org/10.1063/5.0118200), 
X. Zhang, G. K.-L. Chan, *J. Chem. Phys.*, **157**, 204801 (2022)
