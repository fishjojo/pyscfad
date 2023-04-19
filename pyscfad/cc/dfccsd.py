from pyscf import numpy as np
from pyscfad import util
from pyscfad.df.addons import restore
from pyscfad.cc import ccsd, rccsd

CC_Tracers = ['_scf', 'mol', 'with_df']

@util.pytree_node(CC_Tracers, num_args=1)
class RCCSD(rccsd.RCCSD):
    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None, **kwargs):
        super().__init__(mf, frozen=frozen, mo_coeff=mo_coeff, mo_occ=mo_occ,
                       **kwargs)
        if getattr(mf, 'with_df', None):
            self.with_df = mf.with_df
        else:
            raise KeyError('The mean-field object has no density fitting.')

        self._keys.update(['with_df'])
        self.__dict__.update(kwargs)

    def ao2mo(self, mo_coeff=None):
        return _make_df_eris_incore(self, mo_coeff)


@util.pytree_node(ccsd.ERI_Tracers)
class _ChemistsERIs(rccsd._ChemistsERIs):
    def __init__(self, mol=None, **kwargs):
        super().__init__(mol=mol, **kwargs)
        self.naux = None
        self.vvL = None
        self.__dict__.update(kwargs)


def _make_df_eris_incore(cc, mo_coeff=None):
    eris = _ChemistsERIs()
    eris._common_init_(cc, mo_coeff)
    nocc = eris.nocc
    nmo = eris.fock.shape[0]
    nvir = nmo - nocc
    with_df = cc.with_df
    naux = with_df.get_naoaux()

    mo = np.asarray(eris.mo_coeff)
    nao = mo.shape[0]
    orbo = mo[:,:nocc]
    orbv = mo[:,nocc:]

    Lpq = restore('s1', with_df._cderi, nao)
    Loo = np.einsum('lpq,pi,qj->lij', Lpq, orbo, orbo).reshape(naux,-1)
    Lov = np.einsum('lpq,pi,qa->lia', Lpq, orbo, orbv).reshape(naux,-1)
    Lvv = np.einsum('lpq,pa,qb->lab', Lpq, orbv, orbv).reshape(naux,-1)

    eris.oooo = np.dot(Loo.T, Loo).reshape(nocc,nocc,nocc,nocc)
    eris.ovoo = np.dot(Lov.T, Loo).reshape(nocc,nvir,nocc,nocc)
    ovov = np.dot(Lov.T, Lov).reshape(nocc,nvir,nocc,nvir)
    eris.ovov = ovov
    eris.ovvo = ovov.transpose(0,1,3,2)

    eris.oovv = np.dot(Loo.T, Lvv).reshape(nocc,nocc,nvir,nvir)
    eris.ovvv = np.dot(Lov.T, Lvv).reshape(nocc,nvir,nvir,nvir)
    eris.vvvv = np.dot(Lvv.T, Lvv).reshape(nvir,nvir,nvir,nvir)
    return eris
