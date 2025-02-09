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

# Orbital optimization

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fishjojo/pyscfad/blob/doc/doc/source/getting_started/tutorial/02_oorpa.ipynb)

Another example of using pyscfad is to apply orbital optimization for complex quantum chemistry methods.
Here, we present an implementation of the orbital optimized random phase approximation (OO-RPA) method.

+++

## OO-RPA

First, a reference Kohn-Sham DFT calculation is performed.

```{code-cell} ipython3
from pyscfad import gto, dft

mol = gto.Mole()
mol.atom = [['He', (0., 0., 0.)],
            ['He', (0., 0., 2.6)]]
mol.basis = 'def2-svp'
mol.verbose = 0
mol.build()

mf = dft.RKS(mol, xc='PBE')
e_pbe = mf.kernel()
print(f'PBE energy (in Eh): {e_pbe}')
```

Then, one needs to define the energy function for the RPA method, with the orbital rotation matrix as the variable.

```{code-cell} ipython3
from pyscf.df.addons import make_auxbasis
from pyscfad import df
from pyscfad.gw import rpa
from pyscfad.tools import rotate_mo1

# initial MO coefficients
mo0 = mf.mo_coeff
# density fitting object
mydf = df.DF(mol, make_auxbasis(mol, mp2fit=True))

def rpa_energy(x):
    # apply orbital rotation
    mf.mo_coeff = rotate_mo1(mo0, x)
    # density-fitted RPA
    myrpa = rpa.RPA(mf)
    myrpa.with_df = mydf
    myrpa.kernel()
    return myrpa.e_tot
```

Here, we use the differentiable [RPA method](https://github.com/fishjojo/pyscfad/blob/main/pyscfad/gw/rpa.py) implemented in pyscfad.
And [density fitting](https://github.com/fishjojo/pyscfad/blob/main/pyscfad/df/df.py) is enabled as well.
The function `rotate_mo1` applies the unitary orbital rotation to the MOs.
Note that the DFT object and the density fitting object are kept fixed when computing the energy,
and thus can be constructed outside of the energy function.

+++

The analytoc Jacobian and Hessian of the energy are conveniently defined using the jax built-in functions.

```{code-cell} ipython3
import jax

# jacobian
jac = lambda x, *args: jax.jacrev(rpa_energy)(x)
# hessian vector product
hessp = lambda x, p, *args: jax.vjp(jac, x)[1](p)[0]
```

Finally, the energy can be minimized by conventional optimizers, e.g., those provided by scipy.

```{code-cell} ipython3
import numpy
from scipy.optimize import minimize

x0 = numpy.zeros([mol.nao*(mol.nao-1)//2,])
res = minimize(rpa_energy, x0, jac=jac, hessp=hessp,
               method='trust-krylov', options={'gtol': 1e-6})
print(f'OO-RPA/PBE energy: {rpa_energy(res.x)}')
```
