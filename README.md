<div align="left">
  <img src="https://fishjojo.github.io/pyscfad/_static/pyscfad_logo.svg" height="80px"/>
</div>

PySCF with Auto-differentiation
===============================

[![PyPI version](https://img.shields.io/pypi/v/pyscfad.svg)](https://pypi.org/project/pyscfad/)
[![Python versions](https://img.shields.io/pypi/pyversions/pyscfad.svg)](https://pypi.org/project/pyscfad/)
[![Build Status](https://github.com/fishjojo/pyscfad/workflows/CI/badge.svg)](https://github.com/fishjojo/pyscfad/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/fishjojo/pyscfad/branch/main/graph/badge.svg?token=NLSWGI0PLE)](https://codecov.io/gh/fishjojo/pyscfad)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6960749.svg)](https://doi.org/10.5281/zenodo.6960749)

* [Documentation](https://fishjojo.github.io/pyscfad/index.html)

Installation
------------

To install the latest release, choose the command matching your hardware:

| Hardware                | Installation                          |
|-------------------------|---------------------------------------|
| CPU                     | `pip install pyscfad`                 |
| NVIDIA GPU (CUDA 12)    | `pip install "pyscfad[cuda12]"`       |
| NVIDIA GPU (CUDA 13)    | `pip install "pyscfad[cuda13]"`       |

To install the development version, run:
```
pip install git+https://github.com/fishjojo/pyscfad.git
```
The dependent C/C++ library `pyscfadlib` can be compiled from source following the instruction
[here](https://fishjojo.github.io/pyscfad/getting_started/install.html#installing-pyscfadlib).

### Supported platforms

Prebuilt wheels are published for the following platforms:

| Platform                      | CPU | NVIDIA GPU |
|-------------------------------|-----|------------|
| Linux, x86_64                 | yes | yes        |
| Linux, aarch64                | yes | yes        |
| macOS, Apple silicon (arm64)  | yes | n/a        |
| macOS, Intel (x86_64)         | no  | n/a        |
| Windows, x86_64               | no  | no         |
| Windows WSL2, x86_64          | yes | yes        |

On platforms without a prebuilt wheel, `pyscfadlib` can still be compiled from source
(see the link above).


`pyscfad` depends on
`jax>=0.9.1,<0.11`,
`pyscfadlib>=0.3`,
`pyscf>=2.3`, and
`pyscf-properties`.

Citing PySCFAD
--------------
The following paper should be cited in publications utilizing the PySCFAD program package:

[Differentiable quantum chemistry with PySCF for molecules and materials at the mean-field level and beyond](https://doi.org/10.1063/5.0118200), 
X. Zhang, G. K.-L. Chan, *J. Chem. Phys.*, **157**, 204801 (2022)
