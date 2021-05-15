from typing import Union, Any
import jax
from pyscf import __config__
from pyscf.mp import mp2
from pyscfad import lib, gto
from pyscfad.lib import numpy as jnp
from pyscfad.scf import hf

@lib.dataclass
class MP2(mp2.MP2):
    _scf: hf.SCF = lib.field(pytree_node=True)

    max_cycle: int = getattr(__config__, 'cc_ccsd_CCSD_max_cycle', 50)
    conv_tol: float = getattr(__config__, 'cc_ccsd_CCSD_conv_tol', 1e-7)
    conv_tol_normt: float = getattr(__config__, 'cc_ccsd_CCSD_conv_tol_normt', 1e-5)


    frozen: Union[int, list, None] = None
    level_shift: float = 0.

    mol: gto.Mole = None
    verbose: int = None
    stdout: Any = None
    max_memory: int = None

    mo_coeff: jnp.array = None
    mo_occ: jnp.array = None
    _nocc: int = None
    _nmo: int = None
    e_corr: float = None
    e_hf: float = None
    t2: jnp.array = None

    def __post_init__(self):
        if self.mol is None:
            self.mol = self._scf.mol
        if self.verbose is None:
            self.verbose = self.mol.verbose
        if self.stdout is None:
            self.stdout = self.mol.stdout
        if self.max_memory is None:
            self.max_memory = self._scf.max_memory
        if self.mo_coeff is None:
            self.mo_coeff = self._scf.mo_coeff
        if self.mo_occ is None:
            self.mo_occ = self._scf.mo_occ

        self._keys = set(self.__dict__.keys())

    def ao2mo(self, mo_coeff=None):
        eris = mp2._ChemistsERIs()
        eris._common_init_(self, mo_coeff)
        mo_coeff = eris.mo_coeff

        nocc = self.nocc
        co = jnp.asarray(mo_coeff[:,:nocc])
        cv = jnp.asarray(mo_coeff[:,nocc:])
        eris.ovov = jnp.einsum("uvst,ui,va,sj,tb->iajb", self._scf._eri, co,cv,co,cv)
        return eris

    def mol_grad_ad(self, mode="rev"):
        """
        Energy gradient wrt AO parameters computed by AD
        """
        def e_tot(mymp):
            mymp.reset()
            dm0 = mymp._scf.make_rdm1()
            mymp._scf.kernel(dm0=dm0)
            mymp.mo_coeff = mymp._scf.mo_coeff
            mymp.mo_occ = mymp._scf.mo_occ
            mymp.kernel()
            return mymp.e_tot
        if mode == "rev":
            jac = jax.jacrev(e_tot)(self)
        else:
            jac = jax.jacfwd(e_tot)(self)
        return jac._scf.mol
