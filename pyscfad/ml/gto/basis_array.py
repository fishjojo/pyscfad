# Copyright 2021-2025 The PySCFAD Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import dataclasses
import numpy
import jax
from jax import numpy as jnp

from pyscf.lib.exceptions import BasisNotFoundError
from pyscf.gto import basis as pyscf_basis
from pyscf.gto.mole import BAS_SLOTS, NORMALIZE_GTO
from pyscf.data.elements import _symbol

from pyscfad.gto.mole_lite import _parse_default_basis, _format_basis

# NOTE Monkey patch
def _load_external(module, filename_or_basisname, symb, **kwargs):
    try:
        return module.load(filename_or_basisname, symb, **kwargs)
    except BasisNotFoundError:
        return [[0, [0., 0.],],]
pyscf_basis._load_external = _load_external

@jax.tree_util.register_dataclass
@dataclasses.dataclass
class BasisArray:
    """Basis set stored in an array, padded to make
    each element type have the same numbers of shells,
    primitives and contractions.
    """
    data: jax.Array
    mask_shl: jax.Array
    mask_ctr: jax.Array
    ls: numpy.ndarray = dataclasses.field(metadata=dict(static=True))
    l_loc: numpy.ndarray = dataclasses.field(metadata=dict(static=True))

    def make_bas_env(self, ptr: int=0):
        return make_bas_env(self, ptr=ptr)

    def make_loc(self, natm, key):
        return make_loc(self, natm, key)

    def aoslice_by_atom(self, natm, ao_loc=None, cart=False):
        return aoslice_by_atom(self, natm, ao_loc=ao_loc, cart=cart)

    def make_ao_mask(self, mask_shl, mask_ctr, cart=False):
        return make_ao_mask(self, mask_shl=mask_shl, mask_ctr=mask_ctr, cart=cart)

    def nao_nr(self, cart=False):
        """Number of atomic orbitals per element (non-relativistic).
        """
        ls = self.ls
        if cart:
            return numpy.sum((ls+1)*(ls+2)//2, dtype=numpy.int32)
        else:
            return numpy.sum(2*ls+1, dtype=numpy.int32)

    @property
    def nbas(self):
        """Number of shells per element.
        """
        return len(self.ls)


def gaussian_int(n, alpha):
    from jax.scipy.special import gamma
    n1 = (n + 1) * .5
    return jnp.where(
        jnp.greater(alpha, 1e-12),
        gamma(n1) / (2. * alpha**n1),
        jnp.array(0.),
    )

def gto_norm(l, expnt):
    assert numpy.all(l >= 0)
    return jnp.where(
        jnp.greater(expnt, 1e-12),
        1. / jnp.sqrt(gaussian_int(l*2+2, 2*expnt)),
        jnp.array(0.),
    )

def _nomalize_contracted_ao(l, es, cs):
    ee = es.reshape(-1,1) + es.reshape(1,-1)
    ee = jnp.where(
        jnp.greater(ee, 1e-12),
        gaussian_int(l*2+2, ee),
        jnp.array(0.),
    )
    s1 = 1. / jnp.sqrt(jnp.einsum("pi,pq,qi->i", cs, ee, cs))
    return jnp.einsum("pi,i->pi", cs, s1)

def make_bas_env(
    basis: BasisArray,
    ptr: int = 0,
) -> tuple[jax.Array, jax.Array]:
    _bas = []
    _env = []
    # TODO kappa
    kappa = 0

    data = basis.data
    ls = basis.ls
    for z in range(data.shape[0]):
        basis_add = data[z]
        for i, l in enumerate(ls):
            param = basis_add[i]
            es = param[:,0]
            cs = param[:,1:]
            nprim, nctr = cs.shape

            cs = jnp.where(
                jnp.greater(es, 1e-12)[:,None],
                jnp.einsum("pi,p->pi", cs, gto_norm(l, es)),
                jnp.array(0.),
            )
            if NORMALIZE_GTO:
                cs = jnp.where(
                    jnp.greater(es, 1e-12)[:,None],
                    _nomalize_contracted_ao(l, es, cs),
                    jnp.array(0.),
                )

            _env.append(es)
            _env.append(cs.T.ravel())
            ptr_exp = ptr
            ptr_coeff = ptr_exp + nprim
            ptr = ptr_coeff + nprim * nctr
            _bas.append([0, l, nprim, nctr, kappa, ptr_exp, ptr_coeff, 0])

    _bas = jnp.asarray(_bas, dtype=jnp.int32).reshape(data.shape[0], len(ls), BAS_SLOTS)
    _env = jnp.hstack(_env)
    return _bas, _env

def make_loc(
    basis: BasisArray,
    natm: int,
    key: str,
) -> numpy.ndarray:
    l = numpy.repeat(basis.ls.reshape(1,-1), natm, axis=0).ravel()
    nc = basis.data.shape[-1] - 1
    if "cart" in key:
        dims = (l+1)*(l+2)//2 * nc
    elif "sph" in key:
        dims = (l*2+1) * nc
    else:
        raise NotImplementedError

    ao_loc = numpy.zeros(len(dims)+1, dtype=numpy.int32)
    ao_loc[1:] = numpy.cumsum(dims, dtype=numpy.int32)
    return ao_loc

def aoslice_by_atom(
    basis: BasisArray,
    natm: int,
    ao_loc: numpy.ndarray | None = None,
    cart: bool = False,
) -> numpy.ndarray:
    if ao_loc is None:
        key = "cart" if cart else "sph"
        ao_loc = basis.make_loc(natm, key)

    nshls = numpy.repeat(len(basis.ls), natm)
    shl_loc = numpy.append(0, numpy.cumsum(nshls, dtype=numpy.int32))
    shl_start = shl_loc[:-1]
    shl_end = shl_loc[1:]

    aorange = numpy.empty((natm, 4), dtype=numpy.int32)
    aorange[:,0] = shl_start
    aorange[:,1] = shl_end
    aorange[:,2] = ao_loc[shl_start]
    aorange[:,3] = ao_loc[shl_end]
    return aorange

def make_ao_mask(
    basis: BasisArray,
    mask_shl: jnp.Array,
    mask_ctr: jnp.Array,
    cart: bool = False,
) -> jax.Array:
    ls = basis.ls
    mask_shl_ctr = jnp.einsum("zs,zc->zsc", mask_shl, mask_ctr)

    def _scatter_sph(mask):
        out = [jnp.repeat(mask[i], l * 2 + 1) for i, l in enumerate(ls)]
        return jnp.hstack(out)

    def _scatter_cart(mask):
        out = [jnp.repeat(mask[i], (l+1)*(l+2)//2) for i, l in enumerate(ls)]
        return jnp.hstack(out)

    if not cart:
        return jax.vmap(_scatter_sph)(mask_shl_ctr).ravel()
    else:
        return jax.vmap(_scatter_cart)(mask_shl_ctr).ravel()

def make_basis_array(
    basis: str | dict,
    max_number: int = 118,
) -> BasisArray:
    """Construct a padded array to represent a GTO basis set.

    Parameters
    ----------
    basis : Raw basis set.
    max_number : Maximum atomic number.
    """
    if isinstance(basis, str):
        symbols = [_symbol(z) for z in range(max_number+1)]
        _basis = _parse_default_basis(basis, symbols)
        basis = _format_basis(_basis, symbols)

    max_l = 0
    max_nexp = 0
    max_nc1 = 1
    max_nshl = {}
    for z in range(max_number+1):
        symb = _symbol(z)
        basdic = basis.get(symb)
        for l, bas in basdic.items():
            max_l = max(max_l, l)
            nshl = len(bas)
            if l in max_nshl:
                max_nshl[l] = max(max_nshl[l], nshl)
            else:
                max_nshl[l] = nshl
            for b in bas:
                nexp, nc1 = b.shape
                max_nexp = max(max_nexp, nexp)
                max_nc1 = max(max_nc1, nc1)

    max_nshl = dict(sorted(max_nshl.items()))

    ls = []
    for l, n in max_nshl.items():
        ls += [l,]*n
    ls = numpy.asarray(ls, dtype=numpy.int32)
    l_loc = numpy.append(numpy.array(0, dtype=numpy.int32),
                         numpy.cumsum(numpy.bincount(ls), dtype=numpy.int32))

    a = numpy.zeros([max_number+1, len(ls), max_nexp, max_nc1])
    a[:,:,:,0] = 1e12 # preset exponents to a large number
    mask_shl = numpy.zeros([max_number+1, len(ls)], dtype=bool)
    mask_ctr = numpy.zeros([max_number+1, max_nc1-1], dtype=bool)
    for z in range(max_number+1):
        if z == 0: # dummy atom
            continue
        symb = _symbol(z)
        basdic = basis.get(symb)
        for l, bas in basdic.items():
            for i, b in enumerate(bas):
                l_idx = l_loc[l] + i
                nexp, nc1 = b.shape
                a[z, l_idx, :nexp, :nc1] = b
                mask_shl[z, l_idx] = True
                mask_ctr[z, :nc1-1] = True

    return BasisArray(data=jnp.asarray(a),
                      mask_shl=jnp.asarray(mask_shl),
                      mask_ctr=jnp.asarray(mask_ctr),
                      ls=ls, l_loc=l_loc)


if __name__ == "__main__":
    from pyscfad.xtb import basis as xtb_basis

    basis = xtb_basis.get_basis_filename()
    b = make_basis_array(basis, max_number=9)

    @jax.jit
    def foo(b, idx):
        data = b.data[idx]
        mask_shl = b.mask_shl[idx]
        mask_ctr = b.mask_ctr[idx]
        mask_ao = b.make_ao_mask(mask_shl, mask_ctr)
        return data, mask_shl, mask_ctr, mask_ao

    data, mask_shl, mask_ctr, mask_ao = foo(b, jnp.array([8,1,1,0]))
    print(mask_shl)
    print(mask_ctr)
    print(mask_ao)
