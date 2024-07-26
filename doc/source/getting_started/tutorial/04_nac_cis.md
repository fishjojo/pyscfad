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

# Derivative couplings

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fishjojo/pyscfad/blob/doc/doc/source/getting_started/tutorial/04_nac_cis.ipynb)

The first-order derivative coupling, defined as $\langle \Psi_I |\nabla_\mathbf{R} | \Psi_J\rangle$,
is useful for studying excited state nonadiabatic dynamics.
With automatic differentiation, this quantity can be easily computed.
The major ingradient that needs to be implemented is the overlap between the two wavefunctions $\langle \Psi_I | \Psi_J\rangle$.
In the following, we give an example of the CIS method.

+++

## CIS derivative couplings

First, we need to compute the unperturbed bra wavefunction $\langle \Psi_I |$.

```{code-cell} ipython3
import numpy
import jax
from pyscfad import gto, scf
from pyscfad.tdscf.rhf import CIS, cis_ovlp

mol = gto.Mole()
mol.atom = 'H 0 0 0; H 0 0 1.1'
mol.basis = 'cc-pvdz'
mol.verbose = 0
mol.build(trace_exp=False, trace_ctr_coeff=False)

# HF and CIS calculations
mf = scf.RHF(mol)
mf.kernel()
mytd = CIS(mf)
mytd.nstates = 4
e, x = mytd.kernel()

# CI coefficients of state I
stateI = 0 # the first excited state
xi = x[stateI][0] * numpy.sqrt(2.)
```

Next, we define the function to compute the overlap.
Note that the same CIS calculation is performed to trace the perturbation to the ket wavefunction $|\Psi_J\rangle$.
In addition, the variables corresponding to the unperturbed state is closed over.

```{code-cell} ipython3
def ovlp(mol1):
    mf1 = scf.RHF(mol1)
    mf1.kernel()
    mytd1 = CIS(mf1)
    mytd1.nstates = 4
    _, x1 = mytd1.kernel()
    
    # CI coefficients of state J
    stateJ = 2 # the third excited state
    xj = x1[stateJ][0] * numpy.sqrt(2.)
    
    # CIS wavefunction overlap
    nmo = mf1.mo_coeff.shape[-1]
    nocc = mol1.nelectron // 2
    s = cis_ovlp(mol, mol1, mf.mo_coeff, mf1.mo_coeff,
                 nocc, nocc, nmo, nmo, xi, xj)
    return s
```

Finally, the derivative coupling is computed by differentiating the overlap function.

```{code-cell} ipython3
# Only the ket state is differentiated
mol1 = mol.copy()
nac = jax.grad(ovlp)(mol1).coords
print(f"CIS derivative coupling:\n{nac}")
```
