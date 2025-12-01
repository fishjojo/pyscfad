<div align="left">
  <img src="https://fishjojo.github.io/pyscfad/_static/pyscfad_logo.svg" height="80px"/>
</div>

PySCF with Auto-differentiation
===============================

[![Build Status](https://github.com/fishjojo/pyscfad/workflows/CI/badge.svg)](https://github.com/fishjojo/pyscfad/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/fishjojo/pyscfad/branch/main/graph/badge.svg?token=NLSWGI0PLE)](https://codecov.io/gh/fishjojo/pyscfad)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6960749.svg)](https://doi.org/10.5281/zenodo.6960749)

* [Documentation](https://fishjojo.github.io/pyscfad/index.html)

Installation
------------

* To install the latest release, run:
```
pip install pyscfad
```

* To install the CUDA compatible version, run:
```
pip install pyscfad[cuda12]
```

* To install the development version, run:
```
pip install git+https://github.com/fishjojo/pyscfad.git
```
The dependent C library `pyscfadlib` can be compiled from source following the instruction
[here](https://fishjojo.github.io/pyscfad/getting_started/install.html#installing-pyscfadlib).


`pyscfad` depends on
`numpy`, `scipy`,
`pyscf>=2.3.0`,
`pyscfadlib>=0.1.11`, and
`jax>=0.7.0,<0.8`.

Citing PySCFAD
--------------
The following paper should be cited in publications utilizing the PySCFAD program package:

[Differentiable quantum chemistry with PySCF for molecules and materials at the mean-field level and beyond](https://doi.org/10.1063/5.0118200), 
X. Zhang, G. K.-L. Chan, *J. Chem. Phys.*, **157**, 204801 (2022)
