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

# Response properties

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fishjojo/pyscfad/blob/doc/doc/source/getting_started/tutorial/03_raman_ccsd.ipynb)

Most static response properties can be formulated as the derivative of energy/Lagrangian w.r.t the perturbation (e.g., electric or magnetic fields).
In order to computed these properties, one usually needs to include both the orbital response and the amplitude response,
which can be tedious to implement for complex quantum chemistry methods.
On the contrary, automatic differentiation greatly simplifies these calculations,
where only the energy function needs to be explicitly implemented.
Here, we take the Raman activity as an example to showcase the use of pyscfad in property calculations.

+++

## Raman activity

As usual, we first build the `Mole` object.
And we don't need derivatives w.r.t the basis function exponents and contraction coefficients,
so we turn off the tracing of them.

```{code-cell} ipython3
from pyscfad import gto

mol = gto.Mole()
mol.atom = '''B  ,  0.   0.   0.
              H  ,  0.   0.   2.36328'''
mol.basis = 'aug-cc-pvdz'
mol.unit = 'B'
mol.verbose = 0
# do not trace mol.exp and mol.ctr_coeff
mol.build(trace_exp=False, trace_ctr_coeff=False)
```

Next, we define our energy function. We perform a CCSD calculation with an external electric field applied.
pyscfad provides a differentiable implementation of CCSD, which is used here.

```{code-cell} ipython3
from jax import numpy as np
from pyscfad import scf, cc

# CCSD energy
def energy(mol):
    mf = scf.RHF(mol)
    mf.kernel()
    mycc = cc.RCCSD(mf)
    mycc.kernel()
    return mycc.e_tot

# CCSD energy with external electric field applied
def apply_E(mol, E):
    field = np.einsum('x,xij->ij', E, mol.intor('int1e_r'))
    mf = scf.RHF(mol)
    h1 = mf.get_hcore() + field
    mf.get_hcore = lambda *args, **kwargs: h1
    mf.kernel()
    mycc = cc.RCCSD(mf)
    mycc.kernel()
    return mycc.e_tot
```

In order to compute the Raman activity, we need two ingradients,
namely, the nuclear Hessian ($\frac{d^2 e_{tot}}{d\mathbf{R}^2}$) and
the Raman tensor ($\chi=-\frac{d^3 e_{tot}}{d\mathbf{R} d\mathbf{E}^2}$).

+++

```{note} 
The nuclear Hessian is computed using an energy function without applying the electric field.
This is due to the limitation that the 2nd order nuclear derivative of the integral $\langle a|\mathbf{r}|b \rangle$ is not available.
```

```{code-cell} ipython3
import jax

E0 = np.zeros((3)) # a zero static electric field

hess = jax.jacfwd(jax.grad(energy))(mol).coords.coords
chi = -jax.jacfwd(jax.jacfwd(jax.grad(apply_E, 1), 1), 0)(mol, E0).coords
```

Finally, we compute the Raman activity and the depolarization ration with the `harmonic_analysis` function.

```{code-cell} ipython3
from pyscfad.prop.thermo import vib

vibration, _, raman = vib.harmonic_analysis(mol, hess, raman_tensor=chi)
print("Vibrational frequency in cm^-1:")
print(vibration['freq_wavenumber'])
print('Raman activity in A^4/amu:')
print(raman['activity'])
print('Depolarization ration:')
print(raman['depolar_ratio'])
```
