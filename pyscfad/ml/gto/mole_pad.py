# Copyright 2025-2026 The PySCFAD Authors
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
from typing import TYPE_CHECKING
import warnings

import numpy
from pyscf.gto.mole import (
    ATOM_OF,
    ATM_SLOTS,
    BAS_SLOTS,
    CHARGE_OF,
    NUC_MOD_OF,
    NUC_POINT,
    PTR_COMMON_ORIG,
    PTR_RINV_ORIG,
    PTR_COORD,
    PTR_ENV_START,
    PTR_ZETA,
)

from pyscfad import numpy as np
from pyscfad import ops
from pyscfad.gto import moleintor_lite
from pyscfad.gto import MoleLite
from pyscfad.experimental import moleintor_cuint

if TYPE_CHECKING:
    from pyscfad.typing import ArrayLike, Array
    from pyscfad.ml.gto.basis_array import BasisArray

def tot_electrons(mol: MolePad) -> Array:
    nelectron = mol.atom_charges().sum()
    nelectron -= mol.charge
    nelectron_int = np.round(nelectron).astype(np.int32)
    return nelectron_int


class MolePad(MoleLite):
    """Molecular information with padding (for batched calculations).

    Parameters:
        numbers: Atomic numbers.
        coords: Atomic coordinates (in Bohr).
        basis: Atom-centered contracted Gaussian basis set parameters
            (including exponents and contraction coefficients).
        charge: Total charge.
        spin: 2S (number of alpha electrons minus number of beta electrons).
        cart: Whether to use Cartesian Gaussian basis.
    """
    def __init__(
        self,
        numbers: ArrayLike,
        coords: ArrayLike,
        basis: BasisArray | None = None,
        charge: int = 0,
        spin: int = 0,
        cart: bool = False,
        verbose: int = 3,
        cuint_plan: moleintor_cuint.CuintPlan | None = None,
        **kwargs,
    ):
        if "trace_coords" in kwargs or "trace_basis" in kwargs:
            warnings.warn("'trace_coords' and 'trace_basis' are deprecated. "
                          "Whether derivative is taken w.r.t. a variable is "
                          "dertermined based on JAX tracing only. "
                          "If derivative is not wanted for a variable, "
                          "use jax.stop_gradient.")

        self.numbers = np.asarray(numbers, dtype=np.int32)
        self.coords = np.asarray(coords, dtype=np.floatx)
        self.basis = basis
        self.charge = charge
        self.spin = spin
        self.cart = cart
        self.verbose = verbose
        self.cuint_plan = cuint_plan

        self.atom_mask = np.greater(self.numbers, 0)
        self.shl_mask = None
        self.ao_mask = None

        self._nao = 0
        self._atm = None
        self._bas = None
        self._env = None
        self.r0 = None
        self.exp = None
        self.ctr_coeff = None
        self.common_origin = np.zeros(3, dtype=np.floatx)
        self.rinv_origin = np.zeros(3, dtype=np.floatx)
        if self.basis is not None:
            (self._atm, self._bas, self._env,
             self.r0, self.exp, self.ctr_coeff) = make_env(self)

            self.shl_mask = self.basis.mask_shl[self.numbers].ravel()
            self.ao_mask = self.basis.make_ao_mask(
                self.basis.mask_shl[self.numbers],
                self.basis.mask_ctr[self.numbers],
                cart=self.cart,
            )
            self._nao = None

        self._pseudo = {}
        self._ecpbas = numpy.zeros((0,8), dtype=numpy.int32)
        self._built = True

    def atom_charges(self) -> Array:
        return self.numbers

    def atom_nshells(self, atm_id: int) -> int:
        del atm_id
        return self.basis.nbas

    @property
    def nao(self) -> int:
        if self._nao is None:
            return self.nao_nr()
        else:
            return self._nao

    def nao_nr(self, cart: bool | None = None) -> int:
        if cart is None:
            cart = self.cart
        return self.basis.nao_nr(cart=cart) * self.natm

    @property
    def natm(self) -> int:
        return len(self.numbers)

    def copy(
        self,
        deep: bool = True,
    ) -> MolePad:
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
        out: ArrayLike | None = None,
        shls_slice: tuple[int, ...] | None = None,
        grids: ArrayLike | None = None,
        cuint_plan: moleintor_cuint.CuintPlan | None = None,
    ) -> Array:
        del out, grids

        intor_name = self._add_suffix(intor_name)
        if "ECP" in intor_name:
            raise NotImplementedError
        if "_grids" in intor_name:
            raise NotImplementedError

        origin = None
        if intor_name.startswith("int1e_rinv"):
            origin = self.rinv_origin
        elif intor_name.startswith("int1e_r"):
            origin = self.common_origin

        ao_loc = self.ao_loc

        if cuint_plan is None:
            cuint_plan = self.cuint_plan

        if cuint_plan is not None:
            out = moleintor_cuint.getints(
                intor_name,
                self._atm,
                self._bas,
                self._env,
                cuint_plan,
                self.r0,
                self.exp,
                self.ctr_coeff,
                origin=origin,
                shls_slice=shls_slice,
                comp=comp,
                hermi=hermi,
                aosym=aosym,
                ao_loc=ao_loc,
            )
        else:
            out = moleintor_lite.getints(
                intor_name,
                self._atm,
                self._bas,
                self._env,
                self.r0,
                self.exp,
                self.ctr_coeff,
                origin=origin,
                shls_slice=shls_slice,
                comp=comp,
                hermi=hermi,
                aosym=aosym,
                ao_loc=ao_loc,
                basis_array_metadata=self.basis.metadata,
            )
        return out

    def ao_loc_nr(self) -> numpy.ndarray:
        if self.cart:
            key = "cart"
        else:
            key = "sph"
        return self.basis.make_loc(self.natm, key)

    ao_loc = property(ao_loc_nr)
    ao_loc_2c = NotImplemented

    def aoslice_by_atom(self, ao_loc: ArrayLike | None = None) -> numpy.ndarray:
        if ao_loc is None:
            ao_loc = self.ao_loc
        return self.basis.aoslice_by_atom(self.natm, ao_loc=ao_loc)

    tot_electrons = tot_electrons
    atom_pure_symbol = NotImplemented
    from_pyscf = NotImplemented
    to_pyscf = NotImplemented

def make_atm_env(
    coords: Array,
    numbers: Array,
    ptr: int = 0,
    nuclear_model: int = NUC_POINT,
    nucprop: dict | None = None,
) -> tuple[Array, Array, Array]:
    r0 = coords.reshape(-1, 3)
    natm = r0.shape[0]
    nuc_charge = numbers
    if nuclear_model == NUC_POINT:
        zeta = np.zeros((natm,1), dtype=np.floatx)
    else:
        raise NotImplementedError(f"nuclear_model = {nuclear_model} is not supported")
    env = np.hstack([r0, zeta]).ravel()

    atm = np.zeros((natm, ATM_SLOTS), dtype=np.int32)
    atm = ops.index_update(atm, ops.index[:,CHARGE_OF], nuc_charge)
    atm = ops.index_update(atm, ops.index[:,PTR_COORD],
                           np.arange(ptr, ptr+4*natm, 4, dtype=np.int32))
    atm = ops.index_update(atm, ops.index[:,NUC_MOD_OF],
                           np.array(nuclear_model, dtype=np.int32))
    atm = ops.index_update(atm, ops.index[:,PTR_ZETA],
                           atm[:,PTR_COORD] + np.array(3, dtype=np.int32))
    return atm, env, r0

def make_env(
    mol: MolePad,
) -> tuple[Array, ...]:
    """Make ``_atm``, ``_bas``, and ``_env`` for
    interfacing with libcint.
    """
    pre_env = np.zeros(PTR_ENV_START, dtype=np.floatx)
    pre_env = ops.index_update(
        pre_env,
        ops.index[PTR_COMMON_ORIG:PTR_COMMON_ORIG+3],
        np.asarray(mol.common_origin, dtype=np.floatx),
    )
    pre_env = ops.index_update(
        pre_env,
        ops.index[PTR_RINV_ORIG:PTR_RINV_ORIG+3],
        np.asarray(mol.rinv_origin, dtype=np.floatx),
    )

    _env = [pre_env]
    ptr_env = pre_env.size

    # TODO other nuclear charge models
    _atm, env1, r0 = make_atm_env(mol.coords, mol.numbers, ptr_env)
    _env.append(env1)
    ptr_env += env1.size

    bas0, env0, exp, ctr_coeff = mol.basis.make_bas_env(ptr_env)

    _bas = bas0[mol.numbers]
    _bas = ops.index_update(_bas, ops.index[:,:,ATOM_OF],
                            np.arange(len(_atm), dtype=np.int32)[:,None])
    _bas = _bas.reshape(-1, BAS_SLOTS)
    _env = np.hstack(_env)
    _env = np.hstack([_env, env0])
    return _atm, _bas, _env, r0, exp, ctr_coeff
