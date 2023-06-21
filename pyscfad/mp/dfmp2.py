from pyscf import __config__
from pyscf import numpy as np
from pyscf.lib import current_memory, logger
from pyscf.mp.mp2 import _ChemistsERIs
from pyscfad import util
from pyscfad.lib import vmap
from pyscfad.ao2mo import _ao2mo
from pyscfad.mp import mp2

WITH_T2 = getattr(__config__, 'mp_dfmp2_with_t2', True)

def kernel(mp, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2,
           verbose=None):
    if mo_energy is not None or mo_coeff is not None:
        assert (mp.frozen == 0 or mp.frozen is None)

    if eris is None:
        eris = mp.ao2mo(mo_coeff)
    if mo_energy is None:
        mo_energy = eris.mo_energy
    if mo_coeff is None:
        mo_coeff = eris.mo_coeff

    nocc = mp.nocc
    nvir = mp.nmo - nocc
    naux = mp.with_df.get_naoaux()
    eia = mo_energy[:nocc,None] - mo_energy[None,nocc:]

    Lov = mp.loop_ao2mo(mo_coeff, nocc)

    def body(Lv, Lov, ea, eia):
        gi = np.einsum('la,ljb->jab', Lv, Lov)
        t2i = gi / (eia[:,:,None] + ea[None,None,:])
        ei = np.einsum('jab,jab', t2i, gi) * 2 - np.einsum('jab,jba', t2i, gi)
        return ei, t2i

    e, t2 = vmap(body, in_axes=(1,None,0,None))(Lov, Lov, eia, eia)
    if not with_t2:
        t2 = None
    emp2 = e.sum().real
    return emp2, t2

@util.pytree_node(['_scf', 'mol', 'with_df'], num_args=1)
class MP2(mp2.MP2):
    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None, **kwargs):
        super().__init__(mf, frozen=frozen,
                         mo_coeff=mo_coeff, mo_occ=mo_occ, **kwargs)
        if getattr(mf, 'with_df', None):
            self.with_df = mf.with_df
        else:
            raise KeyError('The mean-field object has no density fitting.')

        self._keys.update(['with_df'])
        self.__dict__.update(kwargs)

    def ao2mo(self, mo_coeff=None):
        eris = _ChemistsERIs()
        eris._common_init_(self, mo_coeff)
        return eris

    def loop_ao2mo(self, mo_coeff, nocc):
        # NOTE return the whole 3c integral for now
        nao, nmo = mo_coeff.shape
        nvir = nmo - nocc
        ijslice = (0, nocc, nocc, nmo)

        with_df = self.with_df
        naux = with_df.get_naoaux()
        mem_incore = (naux*nocc*nvir + 2*(nocc*nvir)**2) * 8 / 1e6
        mem_now = current_memory()[0]
        if (mem_incore + mem_now < self.max_memory) or self.mol.incore_anyway:
            eri1 = with_df._cderi
            Lov = _ao2mo.nr_e2(eri1, mo_coeff, ijslice, aosym='s2')
            return Lov.reshape((naux, nocc, nvir))
        else:
            raise RuntimeError(f'{mem_incore+mem_now} MB of memory is needed.')

    def kernel(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2):
        if self.verbose >= logger.WARN:
            self.check_sanity()

        self.dump_flags()

        self.e_hf = self.get_e_hf(mo_coeff=mo_coeff)

        if eris is None:
            eris = self.ao2mo(mo_coeff)

        if self._scf.converged:
            self.e_corr, self.t2 = self.init_amps(mo_energy, mo_coeff, eris, with_t2)
        else:
            raise NotImplementedError

        self.e_corr = self.e_corr
        self._finalize()
        return self.e_corr, self.t2

    def init_amps(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2):
        return kernel(self, mo_energy, mo_coeff, eris, with_t2)

del WITH_T2
