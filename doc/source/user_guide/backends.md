---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Backends

The future development of pyscfad may support multiple numpy-like backends
with a universal implementation. The work is still on-going.
As of version 0.1, only the JAX backend is fully tested.
However, it is recommended to call numpy functions
through the `pyscfad.numpy` module.

## Switching backends

The numpy-like backends can be specified with the environment variable
`PYSCFAD_BACKEND`. By default, the JAX backend is used.
In addition, `numpy` and `torch` backends may be specified, e.g.,

```
export PYSCFAD_BACKEND='torch'
```

With the numpy backend, pyscfad would behave like pyscf,
and may be useful when certain methods are not available in pyscf.
The torch backend has limited functionality.
An example of performing Hartree-Fock calculation with input Fock
matrix can be found [here](https://github.com/fishjojo/pyscfad/blob/main/pyscfad/ml/scf/hf.py).

## Numpy

The numpy functions are registered in the `pyscfad.numpy` module,
which is a wrapper to the numpy-like backends.
It is recommended to call numpy functions as follows.

```{code-cell}
from pyscfad import numpy as np

a = np.ones((4,4))
print(type(a))

w, v = np.linalg.eigh(a)
print(w)
```

## Scipy

pyscfad does not provide a scipy wrapper at the moment.
However, the `pyscfad.scipy` module contains some custom
scipy functions that may be useful.
For example, `pyscfad.scipy.linalg.eigh` extends
[`jax.scipy.linalg.eigh`](https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.linalg.eigh.html)
to allow for differentiable generalized eigen decompositions.
Similarly, `pyscfad.scipy.linalg.svd` extends
[`jax.scipy.linalg.svd`](https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.linalg.svd.html)
to allow for differentiation when returning the full matrix.

## Other operations

The `pyscfad.ops` module provides useful operations,
most of which are wrappers to JAX functions that are compatible with other backends.
For instance, it contains `jit`, `vmap`, `stop_gradient`, etc.

