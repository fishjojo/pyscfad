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
from typing import Any

import numpy
from pyscf.gto.mole import (
    ATOM_OF,
    ATM_SLOTS,
    BAS_SLOTS,
    CHARGE_OF,
    NUC_MOD_OF,
    NUC_POINT,
    PTR_COORD,
    PTR_ENV_START,
    PTR_ZETA,
    PTR_EXP,
    PTR_COEFF,
)

from pyscfad import numpy as np
from pyscfad import ops
from pyscfad.gto.mole_lite import MoleLite
from pyscfad.ml.gto.basis_array import BasisArray

Array = Any


def tot_electrons(mol):
    nelectron = mol.atom_charges().sum()
    nelectron -= mol.charge
    nelectron_int = np.round(nelectron).astype(np.int32)
    return nelectron_int


class Mole(MoleLite):
    """Molecular information with padding.

    Parameters
    ----------
    numbers : array
        Atomic numbers.
    coords : array
        Atomic coordinates (in Bohr).
    basis : BasisArray
        Atom-centered contracted Gaussian basis set parameters
        (including exponents and contraction coefficients).
    charge : int
        Total charge.
    spin : int
        2S (number of alpha electrons minus number of beta electrons).
    cart : bool
        Whether to use Cartesian Gaussian basis.
    trace_coords : bool
        Whether to trace atomic coordinates for gradient calculations.
    trace_basis : bool
        Whether to trace basis set parameters for gradient calculations.
    """
    def __init__(
        self,
        numbers: Array,
        coords: Array,
        basis: BasisArray | None = None,
        charge: int = 0,
        spin: int = 0,
        cart: bool = False,
        verbose: int = 3,
        trace_coords: bool = False,
        trace_basis: bool = False,
        bas0: Array = None,
        env0: Array = None,
    ):
        self.numbers = np.asarray(numbers, dtype=np.int32)
        self.coords = np.asarray(coords, dtype=np.float64)
        self.basis = basis
        self.charge = charge
        self.spin = spin
        self.cart = cart
        self.verbose = verbose
        self.trace_coords = trace_coords
        self.trace_basis = trace_basis

        self.ao_mask = None
        self._atm = self._bas = self._env = None
        if self.basis is not None:
            self._atm, self._bas, self._env = make_env(self, bas0=bas0, env0=env0)

            self.ao_mask = self.basis.make_ao_mask(
                self.basis.mask_shl[self.numbers],
                self.basis.mask_ctr[self.numbers],
                cart=self.cart,
            )

        self._nao = None
        self._pseudo = {}
        self._ecpbas = numpy.zeros((0,8), dtype=numpy.int32)
        self._built = True

    def atom_charges(self):
        return self.numbers

    @property
    def nao(self):
        if self._nao is None:
            return self.nao_nr()
        else:
            return self._nao

    def nao_nr(self, cart=None):
        if cart is None:
            cart = self.cart
        return self.basis.nao_nr(cart=cart) * self.natm

    @property
    def natm(self):
        return len(self.numbers)

    def copy(
        self,
        deep: bool = True,
    ) -> Mole:
        import copy
        newmol = self.view(self.__class__)
        if not deep:
            return newmol

        newmol.coords = np.copy(self.coords)
        newmol.basis = copy.deepcopy(self.basis)
        newmol._atm = np.copy(self._atm)
        newmol._bas = np.copy(self._bas)
        newmol._env = np.copy(self._env)
        return newmol

    def intor(
        self,
        intor_name: str,
        comp: int | None = None,
        hermi: int = 0,
        aosym: str = "s1",
        out: Array | None = None,
        shls_slice: tuple[int, ...] | None = None,
        grids: Array | None = None,
    ) -> Array:
        from pyscfad.gto import moleintor_lite
        intor_name = self._add_suffix(intor_name)
        if "ECP" in intor_name:
            raise NotImplementedError
        if "_grids" in intor_name:
            raise NotImplementedError

        natm = len(self._atm)
        ao_loc = self.basis.make_loc(natm, intor_name)
        aoslices = self.basis.aoslice_by_atom(natm, ao_loc=ao_loc)[:,2:4]

        out = moleintor_lite.getints(
            intor_name,
            self._atm,
            self._bas,
            self._env,
            shls_slice=shls_slice,
            comp=comp,
            hermi=hermi,
            aosym=aosym,
            ao_loc=ao_loc,
            out=out,
            trace_coords=self.trace_coords,
            trace_basis=self.trace_basis,
            aoslices=aoslices,
        )
        return out

    tot_electrons = tot_electrons
    atom_pure_symbol = NotImplemented
    from_pyscf = NotImplemented
    to_pyscf = NotImplemented

MolePad = Mole

def make_atm_env(
    coords: Array,
    numbers: Array,
    ptr: int = 0,
    nuclear_model: int = NUC_POINT,
    nucprop: dict | None = None,
) -> tuple[Array, Array]:
    natm = len(coords)
    nuc_charge = numbers
    if nuclear_model == NUC_POINT:
        zeta = np.zeros((natm,1))
    else:
        raise NotImplementedError
    _env = np.hstack((coords, zeta)).ravel()

    _atm = np.zeros((natm, ATM_SLOTS), dtype=np.int32)
    _atm = ops.index_update(_atm, ops.index[:,CHARGE_OF],
                            np.asarray(nuc_charge, dtype=np.int32))
    _atm = ops.index_update(_atm, ops.index[:,PTR_COORD],
                            np.arange(ptr, ptr+4*natm, 4, dtype=np.int32))
    _atm = ops.index_update(_atm, ops.index[:,NUC_MOD_OF],
                            np.arange(nuclear_model, dtype=np.int32))
    _atm = ops.index_update(_atm, ops.index[:,PTR_ZETA],
                            _atm[:,PTR_COORD] + np.array(3, dtype=np.int32))
    return _atm, _env

def make_env(
    mol: Mole,
    bas0: Array = None,
    env0: Array = None,
) -> tuple[Array, Array, Array]:
    """Make ``_atm``, ``_bas``, and ``_env`` for
    interfacing with libcint.
    """
    pre_env = np.zeros(PTR_ENV_START)
    _env = [pre_env]
    ptr_env = pre_env.size

    # TODO other nuclear charge models
    _atm, env1 = make_atm_env(mol.coords, mol.numbers, ptr_env)
    _env.append(env1)
    ptr_env += env1.size

    if bas0 is None or env0 is None:
        bas0, env0 = mol.basis.make_bas_env(ptr_env)
    else:
        bas0 = ops.index_add(bas0, ops.index[:,:,PTR_EXP],
                             np.array(ptr_env, dtype=np.int32))
        bas0 = ops.index_add(bas0, ops.index[:,:,PTR_COEFF],
                             np.array(ptr_env, dtype=np.int32))

    _bas = bas0[mol.numbers]
    _bas = ops.index_update(_bas, ops.index[:,:,ATOM_OF],
                            np.arange(len(_atm), dtype=np.int32)[:,None])
    _bas = _bas.reshape(-1, BAS_SLOTS)
    _env = np.hstack(_env)
    _env = np.hstack([_env, env0])
    return _atm, _bas, _env


if __name__ == "__main__":
    import jax
    from pyscfad.xtb import basis as xtb_basis
    from pyscfad.ml.gto.basis_array import basis_array

    basis = basis_array(xtb_basis.get_basis_filename(), max_number=8)

    numbers = np.array([8, 1, 1], dtype=np.int32)
    coords = np.array(
        [
            [0.00000,  0.00000,  0.00000],
            [1.43355,  0.00000, -0.95296],
            [1.43355,  0.00000,  0.95296],
        ]
    )

    def foo(numbers, coords, bas0=None, env0=None):
        mol = MolePad(numbers=numbers, coords=coords, basis=basis,
                      bas0=bas0, env0=env0, trace_coords=True)
        s = mol.intor("int1e_ovlp")
        return np.linalg.norm(s)

    e, g = jax.jit(jax.value_and_grad(foo, 1))(numbers, coords)
    print(e, g)
