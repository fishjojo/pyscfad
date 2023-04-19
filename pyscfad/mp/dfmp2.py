from pyscf import numpy as np
from pyscf.lib import current_memory
from pyscf.mp.mp2 import _ChemistsERIs
from pyscfad import util
from pyscfad.df.addons import restore
from pyscfad.mp import mp2


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
        mo_coeff = eris.mo_coeff

        nao = mo_coeff.shape[0]
        nocc = self.nocc
        nvir = self.nmo - nocc
        co = np.asarray(mo_coeff[:,:nocc])
        cv = np.asarray(mo_coeff[:,nocc:])

        with_df = self.with_df
        naux = with_df.get_naoaux()
        mem_incore = (naux*nao**2 + (nocc*nvir)**2) * 8 / 1e6
        mem_now = current_memory()[0]
        if (mem_incore + mem_now < 0.99 * self.max_memory) or self.mol.incore_anyway:
            Lpq = restore('s1', with_df._cderi, nao)
            Lov = np.einsum('lpq,pi,qa->lia', Lpq, co, cv)
            eris.ovov = np.einsum('lia,ljb->iajb', Lov, Lov)
        else:
            raise RuntimeError(f'{mem_incore+mem_now} MB of memory is needed.')
        return eris
