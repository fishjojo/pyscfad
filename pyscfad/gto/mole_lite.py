from __future__ import annotations
from typing import Any
from functools import partial

import numpy
import pyscf
from pyscf.lib import with_doc
from pyscf.data.elements import charge
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
    NORMALIZE_GTO,
)

from pyscfad import numpy as np
from pyscfad import pytree
from pyscfad import ops
from pyscfad.ops import jit

class Mole(pytree.PytreeNode):
    """Molecular information.

    Attributes
    ----------
    coords : array
        Atomic coordinates (in a.u.).
    atomic_symbols : tuple of str
        Atomic symbols.
    cgto_params : dict
        Atom-centered contracted Gaussian basis function parameters
        (including exponents and contraction coefficients).
    charge : int
        Total charge.
    spin : int
        2S (number of alpha electrons minus number of beta electrons).
    cart : bool
        Whether to use Cartesian Gaussian basis.
    """
    _dynamic_attr = [
        "coords",
        "cgto_params",
    ]

    def __init__(
        self,
        coords,
        atomic_symbols: tuple[str],
        cgto_params: dict | None = None,
        charge: int = 0,
        spin: int = 0,
        cart: bool = False,
    ):
        self.coords = coords
        self.atomic_symbols = atomic_symbols
        self.cgto_params = cgto_params
        self.charge = charge
        self.spin = spin
        self.cart = cart

    @partial(jit, static_argnames=["intor_name", "comp", "hermi", "aosym", "shls_slice"])
    def intor(
        self,
        intor_name: str,
        comp: int | None = None,
        hermi: int = 0,
        aosym: str = "s1",
        shls_slice: tuple[int] | None = None,
        grids: Any | None = None,
    ):
        return 0

    @classmethod
    def from_pyscf(
        cls,
        mol: pyscf.gto.MoleBase,
    ) -> Mole:
        if not mol._built:
            raise KeyError(f"{mol} not built")
        if mol.ecp or mol.pseudo:
            raise NotImplementedError
        coords = np.asarray(mol.atom_coords())
        atomic_symbols = tuple([mol.atom_symbol(a) for a in range(mol.natm)])

        cgto_params = {}
        for k, v in mol._basis.items():
            if k not in set(atomic_symbols):
                continue
            tmp = {}
            for shell in v:
                l = shell[0]
                param = np.asarray(shell[1:])
                tmp.setdefault(l, []).append(param)
            cgto_params[k] = {l: tmp[l] for l in sorted(tmp)}
        for k in set(atomic_symbols):
            if k not in cgto_params:
                raise ValueError(
                    f"Atomic symbol '{k}' not found in the basis set {mol.basis}.\n"
                    f"{cls} requires one-to-one mapping between the input "
                    "atomic symbols and those in the basis set definition."
                )

        dmol = cls(
            coords,
            atomic_symbols,
            cgto_params=cgto_params,
            charge=mol.charge,
            spin=mol.spin,
            cart=mol.cart,
        )
        return dmol

    def to_pyscf(
        self,
        verbose: int | None = None,
        output: str | None = None,
        max_memory: int | None = None,
    ) -> pyscf.gto.Mole:
        coords = ops.to_numpy(self.coords)
        atom = [[a, tuple(x.tolist())] for a, x in zip(self.atomic_symbols, coords)]

        basis = {}
        for k, v in self.cgto_params.items():
            for l, shells in v.items():
                for shl_param in shells:
                    basis.setdefault(k, []).append([l, *(ops.to_numpy(shl_param).tolist())])

        mol = pyscf.M(
            atom=atom,
            basis=basis,
            charge=self.charge,
            spin=self.spin,
            cart=self.cart,
            unit="AU",
            verbose=verbose,
            output=output,
            max_memory=max_memory,
            dump_input=False,
            parse_arg=False,
        )
        return mol


def gaussian_int(
    n: int | numpy.ndarray,
    alpha: Any,
) -> Any:
    r"""Gaussian integral.
    Computes :math:`\int_0^\infty x^n exp(-alpha x^2) dx`.
    """
    from pyscfad.scipy.special import gamma
    n1 = (n + 1) * .5
    return gamma(n1) / (2. * alpha**n1)

@with_doc(pyscf.gto.mole.gto_norm.__doc__)
def gto_norm(
    l: int | numpy.ndarray,
    expnt: Any,
) -> Any:
    assert numpy.all(l >= 0)
    return 1. / np.sqrt(gaussian_int(l*2+2, 2*expnt))

def _nomalize_contracted_ao(l, es, cs):
    ee = es.reshape(-1,1) + es.reshape(1,-1)
    ee = gaussian_int(l*2+2, ee)
    s1 = 1. / np.sqrt(np.einsum("pi,pq,qi->i", cs, ee, cs))
    return np.einsum("pi,i->pi", cs, s1)

def make_atm_env(
    coords,
    atomic_symbols: tuple[str],
    ptr: int = 0,
    nuclear_model: int = NUC_POINT,
    nucprop: dict | None = None,
) -> tuple[numpy.ndarray, Any]:
    natm = len(coords)
    nuc_charge = [charge(symb) for symb in atomic_symbols]
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
) -> tuple[numpy.ndarray, Any]:
    _bas = []
    _env = []
    # TODO kappa
    kappa = 0
    for l, shells in basis_add.items():
        for param in shells:
            es = param[:,0]
            cs = param[:,1:]
            nprim, nctr = cs.shape
            cs = np.einsum('pi,p->pi', cs, gto_norm(l, es))
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

@jit
def make_env(
    mol: Mole,
) -> tuple[numpy.ndarray, numpy.ndarray, Any]:
    """Make ``_atm``, ``_bas``, and ``_env`` for
    interfacing with libcint.
    """
    pre_env = np.zeros(PTR_ENV_START)
    _env = [pre_env]
    ptr_env = pre_env.size

    # TODO other nuclear charge models
    _atm, env0 = make_atm_env(mol.coords, mol.atomic_symbols, ptr_env)
    _env.append(env0)
    ptr_env += env0.size

    _basdic = {}
    for symb, basis_add in mol.cgto_params.items():
        bas0, env0 = make_bas_env(basis_add, 0, ptr_env)
        ptr_env += env0.size
        _basdic[symb] = bas0
        _env.append(env0)

    _bas = []
    for ia, symb in enumerate(mol.atomic_symbols):
        if symb in _basdic:
            b = _basdic[symb].copy()
        else:
            raise RuntimeError(f"Basis for '{symb}' not found")
        b[:,ATOM_OF] = ia
        _bas.append(b)

    _bas = numpy.vstack(_bas)
    _env = np.hstack(_env)
    return _atm, _bas, _env

if __name__ == "__main__":
    import pyscf
    import jax
    pmol = pyscf.M(atom="h1 0 0 0; h2 0 0 1",
                   basis={"H1" : "sto3g", "H2" : "631G**"}, verbose=5)
    mol = Mole.from_pyscf(pmol)
    #mol = mol.to_pyscf()

    def foo(mol):
        _atm, _bas, _env = make_env(mol)
        return np.sum(mol.coords ** 2) + np.linalg.norm(_env**2)

    gfn = jax.jit(jax.grad(foo))
    g = gfn(mol)
    print(g.coords)
    print(g.cgto_params)

    mol.coords = mol.coords.at[0,2].add(0.001)
    mol.cgto_params["H1"][0][0] = mol.cgto_params["H1"][0][0].at[0,0].add(1.2)
    print(mol.coords)
    print(mol.cgto_params)
    g = gfn(mol)
    print(g.coords)
    print(g.cgto_params)
