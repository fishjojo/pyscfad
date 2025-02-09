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

# Simple mean-field calculations

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fishjojo/pyscfad/blob/doc/doc/source/getting_started/tutorial/01_scf.ipynb)

+++

## Energy derivatives w.r.t. molecular parameters

The most straightforward application of pyscfad is to compute energy derivatives w.r.t. the parameters of the `Mole` object.
Currently, three parameters are supported, including nuclear coordinates `Mole.coords`, and exponentes `Mole.exp` and contraction coefficients `Mole.ctr_coeff` of the basis functions. A typical energy derivative calculation involves the following steps.

+++

### 1. Define the Mole object

The `Mole` object constructor follows the same syntax as that of pyscf. In addtion, one can control whether to *trace* (compute the derivatives w.r.t.) the above mentioned parameters. The default is to trace all of them.

```{code-cell} ipython3
from pyscfad import gto
mol = gto.Mole()
mol.atom = "H 0 0 0; H 0 0 0.74"
mol.basis = "6-31G*"
mol.verbose = 0
mol.build(trace_coords=True, trace_exp=True, trace_ctr_coeff=True)
```

### 2. Define the energy function

The energy function takes the `Mole` object as the input, and returns the energy, which is a scalar. In this example, we compute the Hartree-Fock energy.

```{code-cell} ipython3
from pyscfad import scf
def hf_energy(mol):
    mf = scf.RHF(mol)
    ehf = mf.kernel()
    return ehf
```

### 3. Compute the gradient

We use jax as the backend to trace the computational graph and perform the gradient calculation. See e.g., [`jax.value_and_grad`](https://jax.readthedocs.io/en/latest/_autosummary/jax.value_and_grad.html).

```{code-cell} ipython3
import jax
ehf, grad = jax.value_and_grad(hf_energy)(mol)
print(f'RHF energy (in Eh): {ehf}')
```

The gradients w.r.t. each parameter are stored as attributes of `grad`, which is also a `Mole` object.

```{code-cell} ipython3
print(grad)
```

```{code-cell} ipython3
print(f'Nuclear gradient:\n{grad.coords}')
```

```{code-cell} ipython3
print(f'Energy gradient w.r.t. basis function exponents:\n{grad.exp}')
```

```{code-cell} ipython3
print(f'Energy gradient w.r.t. basis function contraction coefficients:\n{grad.ctr_coeff}')
```

### 4. Higher order derivatives

Higher order derivatives can also be computed, although with much higer memory footprint. Two functions,
[`jax.jacfwd`](https://jax.readthedocs.io/en/latest/_autosummary/jax.jacfwd.html) and
[`jax.jacrev`](https://jax.readthedocs.io/en/latest/_autosummary/jax.jacrev.html),
compute the Jacobian with forward- and reverse-mode differentiation, respectively.

```{code-cell} ipython3
hessian = jax.jacfwd(jax.grad(hf_energy))(mol)
print(f'Energy Hessians\n'
      f'∂^2E/∂R^2: {hessian.coords.coords.shape}\n'
      f'∂^2E/∂R∂ε: {hessian.coords.exp.shape}\n'
      f'∂^2E/∂R∂c: {hessian.coords.ctr_coeff.shape}\n')
```

```{note}
Only first-order derivatives w.r.t. `Mole.exp` and `Mole.ctr_coeff` are available at the moment.
```

+++

Third-order derivatives w.r.t. nuclear coordinates can be computed similarly.

```{code-cell} ipython3
third_order_deriv = jax.jacfwd(jax.jacfwd(jax.grad(hf_energy)))(mol)
print(f'∂^3E/∂R^3: {third_order_deriv.coords.coords.coords.shape}')
```
