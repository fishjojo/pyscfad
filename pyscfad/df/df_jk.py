from functools import wraps
from pyscf.df import df_jk as pyscf_df_jk
from pyscfad import config
from pyscfad import numpy as np
from pyscfad import util
from .addons import restore
from ._df_jk_opt import get_jk as get_jk_opt

def get_jk(dfobj, dm, hermi=1, with_j=True, with_k=True, direct_scf_tol=1e-13):
    if config.moleintor_opt:
        return get_jk_opt(dfobj, dm, hermi=hermi,
                          with_j=with_j, with_k=with_k,
                          direct_scf_tol=direct_scf_tol)
    else:
        return get_jk_gen(dfobj, dm, hermi=hermi,
                          with_j=with_j, with_k=with_k,
                          direct_scf_tol=direct_scf_tol)

def get_jk_gen(dfobj, dm, hermi=1, with_j=True, with_k=True, direct_scf_tol=1e-13):
    nao = dfobj.mol.nao
    dms = dm.reshape(-1, nao, nao)
    Lpq = restore('s1', dfobj._cderi, nao)

    vj = vk = 0
    if with_j:
        tmp = np.einsum('Lpq,xpq->xL', Lpq, dms)
        vj = np.einsum('Lpq,xL->xpq', Lpq, tmp)
        vj = vj.reshape(dm.shape)
    if with_k:
        tmp = np.einsum('Lij,xjk->xLki', Lpq, dms)
        vk = np.einsum('Lki,xLkj->xij', Lpq, tmp)
        vk = vk.reshape(dm.shape)
    return vj, vk

@wraps(pyscf_df_jk.density_fit)
def density_fit(mf, auxbasis=None, with_df=None, only_dfj=False):
    # pylint: disable=import-outside-toplevel
    from pyscfad import scf
    from .df import DF
    assert isinstance(mf, scf.hf.SCF)

    if with_df is None:
        with_df = DF(mf.mol)
        with_df.max_memory = mf.max_memory
        with_df.stdout = mf.stdout
        with_df.verbose = mf.verbose
        with_df.auxbasis = auxbasis

    if isinstance(mf, _DFHF):
        if mf.with_df is None:
            mf.with_df = with_df
        elif getattr(mf.with_df, 'auxbasis', None) != auxbasis:
            mf = mf.copy()
            mf.with_df = with_df
            mf.only_dfj = only_dfj
        return mf

    _DFHF.__bases__ = (pyscf_df_jk._DFHF, mf.__class__)
    dfmf = _DFHF(mf, with_df, only_dfj)
    return dfmf

@util.pytree_node(['mol', 'with_df'])
class _DFHF(pyscf_df_jk._DFHF):
    def __init__(self, mf=None, with_df=None, only_dfj=None, **kwargs):
        if mf is not None:
            self.__dict__.update(mf.__dict__)
        self._eri = None
        self.with_df = with_df
        self.only_dfj = only_dfj
        # Unless DF is used only for J matrix, disable direct_scf for K build.
        # It is more efficient to construct K matrix with MO coefficients than
        # the incremental method in direct_scf.
        self.direct_scf = only_dfj
        self.__dict__.update(kwargs)

    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True,
               omega=None):
        if dm is None:
            dm = self.make_rdm1()

        if not self.with_df:
            return super().get_jk(mol, dm, hermi, with_j, with_k, omega)

        with_dfk = with_k and not self.only_dfj

        #TODO GHF
        vj, vk = self.with_df.get_jk(dm, hermi, with_j, with_dfk,
                                     self.direct_scf_tol, omega)
        if with_k and not with_dfk:
            vk = super().get_jk(mol, dm, hermi, False, True, omega)[1]
        return vj, vk

