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

"""
Lightweight :mod:`~pyscfad.gto.mole` module.
"""
from __future__ import annotations
from typing import TYPE_CHECKING
import contextlib

import numpy
import pyscf
from pyscf.lib import with_doc
from pyscf.data.elements import (
    charge as get_charge,
    _atom_symbol,
    _std_symbol,
    _symbol,
)
from pyscf.data.nist import BOHR
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
    NORMALIZE_GTO,
    MoleBase,
    _parse_default_basis,
    format_basis,
    is_au,
)

from pyscfad import numpy as np
from pyscfad import ops
from pyscfad.gto.mole import energy_nuc
from pyscfad.gto import moleintor_lite
from pyscfad.experimental import moleintor_cuint

if TYPE_CHECKING:
    from collections.abc import Callable
    from pyscfad.typing import ArrayLike, Array

def _format_basis(basis, uniq_symbols):
    if isinstance(basis, dict):
        for k, v in basis.items():
            if isinstance(v, dict):
                return basis

    basis = format_basis(basis)
    return _format_basis_from_pyscf(basis, uniq_symbols)

def _format_basis_from_pyscf(pyscf_basis, uniq_symbols):
    basis = {}
    for symb, shls in pyscf_basis.items():
        if symb not in uniq_symbols:
            continue
        tmp = {}
        for shell in shls:
            l = shell[0]
            param = np.asarray(shell[1:])
            tmp.setdefault(l, []).append(param)
        basis[symb] = {l: tmp[l] for l in sorted(tmp)}

    if not basis:
        basis = None
    return basis

def _format_symbols(symbols):
    if symbols is None:
        return symbols
    if isinstance(symbols, str):
        symbols = [symbols,]
    return tuple(_atom_symbol(symb) for symb in symbols)

class MoleLite(MoleBase):
    """Molecular information (lightweight version).

    Parameters:
        symbols: Atomic symbols (mutually exclusive with ``numbers``).
        coords: Atomic coordinates (in Bohr).
        basis: A string indicating the Gaussian basis set to use, or
            a dictionary including the basis set parameters
            (exponents and contraction coefficients).
        numbers: Atomic numbers (mutually exclusive with ``symbols``).
        charge: Total charge.
        spin: 2S (number of alpha electrons minus number of beta electrons).
        cart: Whether to use Cartesian Gaussian basis.
        verbose: Printing level.
        trace_coords: Whether to trace atomic coordinates for gradient calculations.
        trace_basis: Whether to trace basis set parameters for gradient calculations.
        cuint_plan: Plan for using the cuint backend.

    Notes:
        The molecular composition (i.e., ``symbols`` or ``numbers``)
        must be static as input. For dynamic molecular composition,
        refer to :class:`~pyscfad.ml.gto.mole_pad.MolePad`.
    """
    def __init__(
        self,
        symbols: tuple[str, ...] | None = None,
        coords: ArrayLike | None = None,
        basis: dict | str | None = None,
        numbers: tuple[int, ...] | None = None,
        charge: int = 0,
        spin: int = 0,
        cart: bool = False,
        verbose: int = 3,
        trace_coords: bool = False,
        trace_basis: bool = False,
        cuint_plan: moleintor_cuint.CuintPlan | None = None,
    ):
        if numbers is not None:
            if symbols is not None:
                raise KeyError("Only one of 'symbols' and 'numbers' can be specified.")
            self.symbols = tuple(_symbol(i) for i in numbers)
        else:
            self.symbols = _format_symbols(symbols)

        self.coords = coords

        if basis is not None:
            uniq_symbols = set(self.symbols)
            _basis = _parse_default_basis(basis, uniq_symbols)
            basis = _format_basis(_basis, uniq_symbols)
        self.basis = basis

        self.charge = charge
        self.spin = spin
        self.cart = cart
        self.verbose = verbose
        self.trace_coords = trace_coords
        self.trace_basis = trace_basis

        self._pseudo = {}

        self._atm = self._bas = self._env = None
        if self.basis is not None:
            self._atm, self._bas, self._env = make_env(self)
            self._nao = None

        self.cuint_plan = cuint_plan

        self._built = True

    def atom_pure_symbol(
        self,
        atm_id: int,
    ) -> str:
        return _std_symbol(self.symbols[atm_id])

    def atom_coords(
        self,
        unit: str = "Bohr",
    ) -> Array:
        """Atom coordinates.

        Parameters:
            unit: Units of returned coordinates, can be ``Bohr`` or ``Angstrom``.
        """
        if not is_au(unit):
            return self.coords * BOHR
        else:
            return self.coords

    def copy(
        self,
        deep: bool = True,
    ) -> MoleLite:
        import copy
        newmol = self.view(self.__class__)
        if not deep:
            return newmol

        newmol.coords = np.copy(self.coords)
        newmol.basis = copy.deepcopy(self.basis)
        newmol._atm = numpy.copy(self._atm)
        newmol._bas = numpy.copy(self._bas)
        newmol._env = np.copy(self._env)
        return newmol

    def build(self, *args, **kwargs) -> MoleLite:
        """Placeholder for post-init processing.
        Can be used for caching one-time computed quantities.
        """
        return self

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
        """Integral generator.

        Parameters:
            intor_name: Name of the integral.
            comp: Components of the integrals, e.g. ``int1e_ovlp_dr10`` has 3 components.
            hermi: Hermitian symmetry of 1e integrals, can be
                `0` (no symmetry), `1` (hermitian), and `2` (anti-hermitian).
            aosym: Permutation symmetry of 2e integrals, can be
                `s1`, `s2`, `s4`, and `s8`.
            out: Unused.
            shls_slice: Label the start-stop shells for each index in the integral.
                For example, the 8-element tuple for the 2e integral
                tensor ``(ij|kl) = intor('int2e')`` are specified as
                ``(i0, i1, j0, j1, k0, k1, l0, l1)``.
            grids: Unused.
            cuint_plan: Plan for using the cuint backend.

        Returns:
            Computed integral as an array.

        See also:
            :func:`pyscf.gto.Mole.intor`
        """
        del out, grids

        intor_name = self._add_suffix(intor_name)
        if "ECP" in intor_name:
            raise NotImplementedError
        if "_grids" in intor_name:
            raise NotImplementedError

        if cuint_plan is None:
            cuint_plan = self.cuint_plan

        if cuint_plan is not None:
            out = moleintor_cuint.getints(
                intor_name,
                self._atm,
                self._bas,
                self._env,
                cuint_plan,
                shls_slice=shls_slice,
                comp=comp,
                hermi=hermi,
                aosym=aosym,
                trace_coords=self.trace_coords,
                trace_basis=self.trace_basis,
            )
        else:
            out = moleintor_lite.getints(
                intor_name,
                self._atm,
                self._bas,
                self._env,
                shls_slice=shls_slice,
                comp=comp,
                hermi=hermi,
                aosym=aosym,
                trace_coords=self.trace_coords,
                trace_basis=self.trace_basis,
            )
        return out

    def set_common_origin(
        self,
        coord: ArrayLike,
    ) -> MoleLite:
        if self._env is None:
            raise RuntimeError("{self}._env is not initialized, "
                               "possibly because basis is not set.")

        self._env = ops.index_update(
            self._env,
            ops.index[PTR_COMMON_ORIG:PTR_COMMON_ORIG+3],
            coord,
        )
        return self

    def with_common_origin(
        self,
        coord: ArrayLike,
    ):
        coord0 = np.copy(self._env[PTR_COMMON_ORIG:PTR_COMMON_ORIG+3])
        return self._TemporaryMoleContext(self.set_common_origin, (coord,), (coord0,))

    def set_rinv_origin(
        self,
        coord: ArrayLike,
    ) -> MoleLite:
        if self._env is None:
            raise RuntimeError("{self}._env is not initialized, "
                               "possibly because basis is not set.")

        self._env = ops.index_update(
            self._env,
            ops.index[PTR_RINV_ORIG:PTR_RINV_ORIG+3],
            coord,
        )
        return self

    def with_rinv_origin(
        self,
        coord: ArrayLike,
    ):
        coord0 = np.copy(self._env[PTR_RINV_ORIG:PTR_RINV_ORIG+3])
        return self._TemporaryMoleContext(self.set_rinv_origin, (coord,), (coord0,))

    @contextlib.contextmanager
    def _TemporaryMoleContext(
        self,
        method: Callable[...],
        args: tuple[...],
        args_bak: tuple[...],
    ):
        method(*args)
        try:
            yield
        finally:
            method(*args_bak)

    @classmethod
    def from_pyscf(
        cls,
        mol: MoleBase,
        trace_coords: bool = False,
        trace_basis: bool = False,
    ) -> MoleLite:
        """Initialize from the pyscf :class:`~pyscf.gto.mole.Mole` object.
        """
        if not mol._built:
            raise KeyError(f"{mol} not built")
        if mol.ecp or mol.pseudo:
            raise NotImplementedError
        coords = np.asarray(mol.atom_coords())
        symbols = tuple(mol.atom_symbol(i) for i in range(mol.natm))
        uniq_symbols = set(symbols)
        basis = _format_basis_from_pyscf(mol._basis, uniq_symbols)

        #for symb in uniq_symbols:
        #    if symb not in basis:
        #        raise ValueError(
        #            f"Atomic symbol '{symb}' not found in the basis set {mol.basis}.\n"
        #            f"{cls} requires one-to-one mapping between the input "
        #            "atomic symbols and those in the basis set definition."
        #        )

        dmol = cls(
            symbols=symbols,
            coords=coords,
            basis=basis,
            charge=mol.charge,
            spin=mol.spin,
            cart=mol.cart,
            verbose=mol.verbose,
            trace_coords=trace_coords,
            trace_basis=trace_basis,
        )
        return dmol

    def to_pyscf(
        self,
        output: str | None = None,
        max_memory: int | None = None,
    ) -> MoleBase:
        """Convert to the pyscf :class:`~pyscf.gto.mole.Mole` object.
        """
        coords = ops.to_numpy(self.coords)
        atom = [[a, tuple(x.tolist())] for a, x in zip(self.symbols, coords)]

        basis = {}
        for symb, shls_dict in self.basis.items():
            for l, shls in shls_dict.items():
                for shl_param in shls:
                    basis.setdefault(symb, []).append([l, *(ops.to_numpy(shl_param).tolist())])

        mol = pyscf.M(
            atom=atom,
            basis=basis,
            charge=self.charge,
            spin=self.spin,
            cart=self.cart,
            unit="AU",
            verbose=self.verbose,
            output=output,
            max_memory=max_memory,
            dump_input=False,
            parse_arg=False,
        )
        return mol

    energy_nuc = energy_nuc


def gaussian_int(
    n: int | numpy.ndarray,
    alpha: ArrayLike,
) -> Array:
    r"""Gaussian integral.

    Computes :math:`\int_0^\infty x^n exp(-\alpha x^2) dx`.
    """
    from pyscfad.scipy.special import gamma
    n1 = (n + 1) * .5
    return gamma(n1) / (2. * alpha**n1)

@with_doc(pyscf.gto.mole.gto_norm.__doc__)
def gto_norm(
    l: int | numpy.ndarray,
    expnt: ArrayLike,
) -> Array:
    assert numpy.all(l >= 0)
    return 1. / np.sqrt(gaussian_int(l*2+2, 2*expnt))

def _nomalize_contracted_ao(l, es, cs):
    ee = es.reshape(-1,1) + es.reshape(1,-1)
    ee = gaussian_int(l*2+2, ee)
    s1 = 1. / np.sqrt(np.einsum("pi,pq,qi->i", cs, ee, cs))
    return np.einsum("pi,i->pi", cs, s1)

def make_atm_env(
    coords,
    symbols: tuple[str, ...],
    ptr: int = 0,
    nuclear_model: int = NUC_POINT,
    nucprop: dict | None = None,
) -> tuple[numpy.ndarray, Array]:
    natm = len(coords)
    nuc_charge = [get_charge(symb) for symb in symbols]
    if nuclear_model == NUC_POINT:
        zeta = np.zeros((natm,1))
    else:
        raise NotImplementedError
    _env = np.hstack((coords, zeta)).ravel()

    _atm = numpy.zeros((natm, ATM_SLOTS), dtype=numpy.int32)
    _atm[:,CHARGE_OF] = numpy.asarray(nuc_charge, dtype=numpy.int32)
    _atm[:,PTR_COORD] = numpy.arange(ptr, ptr+4*natm, 4, dtype=numpy.int32)
    _atm[:,NUC_MOD_OF] = nuclear_model
    _atm[:,PTR_ZETA] = _atm[:,PTR_COORD] + 3
    return _atm, _env

def make_bas_env(
    basis_add: dict,
    atom_id: int = 0,
    ptr: int = 0,
) -> tuple[numpy.ndarray, Array]:
    _bas = []
    _env = []
    # TODO kappa
    kappa = 0
    for l, shells in basis_add.items():
        for param in shells:
            es = param[:,0]
            cs = param[:,1:]
            nprim, nctr = cs.shape
            cs = np.einsum("pi,p->pi", cs, gto_norm(l, es))
            if NORMALIZE_GTO:
                cs = _nomalize_contracted_ao(l, es, cs)

            _env.append(es)
            _env.append(cs.T.ravel())
            ptr_exp = ptr
            ptr_coeff = ptr_exp + nprim
            ptr = ptr_coeff + nprim * nctr
            _bas.append([atom_id, l, nprim, nctr, kappa, ptr_exp, ptr_coeff, 0])

    _bas = numpy.asarray(_bas, dtype=numpy.int32).reshape(-1, BAS_SLOTS)
    _env = np.hstack(_env)
    return _bas, _env

def make_env(
    mol: MoleLite,
) -> tuple[numpy.ndarray, numpy.ndarray, Array]:
    """Make ``_atm``, ``_bas``, and ``_env`` for
    interfacing with ``libcint``.
    """
    pre_env = np.zeros(PTR_ENV_START)
    _env = [pre_env]
    ptr_env = pre_env.size

    # TODO other nuclear charge models
    _atm, env0 = make_atm_env(mol.coords, mol.symbols, ptr_env)
    _env.append(env0)
    ptr_env += env0.size

    _basdic = {}
    for symb, basis_add in mol.basis.items():
        bas0, env0 = make_bas_env(basis_add, 0, ptr_env)
        ptr_env += env0.size
        _basdic[symb] = bas0
        _env.append(env0)

    _bas = []
    for ia, symb in enumerate(mol.symbols):
        if symb in _basdic:
            b = _basdic[symb].copy()
            b[:,ATOM_OF] = ia
            _bas.append(b)
        #else:
        #    raise RuntimeError(f"Basis for '{symb}' not found")

    _bas = numpy.vstack(_bas)
    _env = np.hstack(_env)
    return _atm, _bas, _env
