from pyscf.lib import square_mat_in_trilu_indices
from pyscfad import numpy as np
from pyscfad import util
from pyscfad import lib
from pyscfad.ao2mo import _ao2mo
from pyscfad.cc import ccsd

CC_Tracers = ['_scf', 'mol', 'with_df']
ERI_Tracers = ['fock', 'mo_energy',
               'oooo', 'ovoo', 'ovov', 'oovv', 'ovvo', 'ovvv', 'Lvv']

@util.pytree_node(CC_Tracers, num_args=1)
class RCCSD(ccsd.CCSD):
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

def _contract_vvvv_t2(mycc, mol, Lvv, t2, out=None, verbose=None):
    '''Ht2 = numpy.einsum('ijcd,acbd->ijab', t2, vvvv)
    '''
    nvir = t2.shape[-1]
    nvir2 = nvir * nvir
    x2 = t2.reshape(-1, nvir2)

    tril2sq = square_mat_in_trilu_indices(nvir)
    tmp = lib.unpack_tril(np.dot(Lvv.T, Lvv))
    tmp1 = tmp[tril2sq].transpose(0,2,1,3).reshape(nvir2,nvir2)
    Ht2tril = np.dot(x2, tmp1)
    tril2sq = None
    return Ht2tril.reshape(t2.shape)

@util.pytree_node(ERI_Tracers)
class _ChemistsERIs(ccsd._ChemistsERIs):
    def __init__(self, mol=None, **kwargs):
        super().__init__(mol=mol, **kwargs)
        self.naux = None
        self.Lvv = None
        self.__dict__.update(kwargs)

    def _contract_vvvv_t2(self, mycc, t2, direct=False, out=None, verbose=None):
        assert not direct
        return _contract_vvvv_t2(mycc, self.mol, self.Lvv, t2)


def _make_df_eris_incore(cc, mo_coeff=None):
    eris = _ChemistsERIs()
    eris._common_init_(cc, mo_coeff)
    nocc = eris.nocc
    nmo = eris.fock.shape[0]
    nvir = nmo - nocc
    with_df = cc.with_df
    naux = with_df.get_naoaux()

    mo = np.asarray(eris.mo_coeff)
    ijslice = (0, nmo, 0, nmo)
    eri1 = with_df._cderi
    # pylint: disable=too-many-function-args
    Lpq = _ao2mo.nr_e2(eri1, mo, ijslice, aosym='s2', mosym='s1').reshape(-1,nmo,nmo)
    Loo = Lpq[:,:nocc,:nocc].reshape(naux,-1)
    Lov = Lpq[:,:nocc,nocc:].reshape(naux,-1)
    eris.Lvv = Lvv = lib.pack_tril(Lpq[:,nocc:,nocc:])

    eris.oooo = np.dot(Loo.T, Loo).reshape(nocc,nocc,nocc,nocc)
    eris.ovoo = np.dot(Lov.T, Loo).reshape(nocc,nvir,nocc,nocc)
    ovov = np.dot(Lov.T, Lov).reshape(nocc,nvir,nocc,nvir)
    eris.ovov = ovov
    eris.ovvo = ovov.transpose(0,1,3,2)

    oovv = np.dot(Loo.T, Lvv)
    eris.oovv = lib.unpack_tril(oovv).reshape(nocc,nocc,nvir,nvir)
    eris.ovvv = np.dot(Lov.T, Lvv).reshape(nocc,nvir,-1)
    return eris
