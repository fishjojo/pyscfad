from pyscf.mp import mp2 as pyscf_mp2
from pyscfad.lib import numpy as jnp

class MP2(pyscf_mp2.MP2):
    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None, **kwargs):
        pyscf_mp2.MP2.__init__(self, mf, frozen=frozen,
                               mo_coeff=mo_coeff, mo_occ=mo_occ)
        self.__dict__.update(kwargs)

    def ao2mo(self, mo_coeff=None):
        eris = pyscf_mp2._ChemistsERIs()
        eris._common_init_(self, mo_coeff)
        mo_coeff = eris.mo_coeff

        nocc = self.nocc
        co = jnp.asarray(mo_coeff[:,:nocc])
        cv = jnp.asarray(mo_coeff[:,nocc:])
        eris.ovov = jnp.einsum("uvst,ui,va,sj,tb->iajb", self._scf._eri, co,cv,co,cv)
        return eris
