from jax import numpy as jnp
from pyscf.df import df_jk as pyscf_df_jk
from pyscfad import lib
from pyscfad import util
from pyscfad import gto
from pyscfad import scf
from pyscfad.df import addons

def density_fit(mf, auxbasis=None, with_df=None, only_dfj=False):
    from pyscfad import df
    assert isinstance(mf, scf.hf.SCF)

    if with_df is None:
        #auxmol has to be defined before creating the DF object,
        #otherwise, its derivative won't be traced
        auxmol = addons.make_auxmol(mf.mol, auxbasis)
        with_df = df.DF(mf.mol, auxmol=auxmol)
        with_df.max_memory = mf.max_memory
        with_df.stdout = mf.stdout
        with_df.verbose = mf.verbose
        with_df.auxbasis = auxbasis

    mf_class = mf.__class__
    kwargs = mf.__dict__.copy()
    mol = kwargs.pop("mol")

    @util.pytree_node(['mol', 'with_df'], num_args=1)
    class DFHF(pyscf_df_jk._DFHF, mf_class):
        def __init__(self, mol, with_df=None, only_dfj=False, **kwargs):
            self.with_df = with_df
            if getattr(self.with_df, "mol", None) is not None:
                self.mol = self.with_df.mol
            else:
                self.mol = mol
            for key in kwargs.keys():
                setattr(self, key, kwargs[key])
            self.only_dfj = only_dfj

            self._eri = None
            self._keys = self._keys.union(['with_df', 'only_dfj'])

        def reset(self, mol=None):
            self.with_df.reset(mol)
            return mf_class.reset(self, mol)

        def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True,
                   omega=None):
            if dm is None: dm = self.make_rdm1()
            if self.with_df and self.only_dfj:
                vj = vk = None
                if with_j:
                    vj, vk = self.with_df.get_jk(dm, hermi, True, False,
                                                 self.direct_scf_tol, omega)
                if with_k:
                    vk = mf_class.get_jk(self, mol, dm, hermi, False, True, omega)[1]
            elif self.with_df:
                vj, vk = self.with_df.get_jk(dm, hermi, with_j, with_k,
                                             self.direct_scf_tol, omega)
            else:
                vj, vk = mf_class.get_jk(self, mol, dm, hermi, with_j, with_k, omega)
            return vj, vk

    return DFHF(mol, with_df, only_dfj, **kwargs)

def get_jk(dfobj, dm, hermi=1, with_j=True, with_k=True, direct_scf_tol=1e-13):
    assert(with_j or with_k)

    nao = dfobj.mol.nao
    dms = dm.reshape(-1, nao, nao)

    vj = vk = 0
    Lpq = dfobj._cderi
    if with_j:
        tmp = jnp.einsum("Lpq,xpq->xL", Lpq, dms)
        vj = jnp.einsum("Lpq,xL->xpq", Lpq, tmp)
        vj = vj.reshape(dm.shape)

    if with_k:
        tmp = jnp.einsum('Lij,xjk->xLki', Lpq, dms)
        vk = jnp.einsum('Lki,xLkj->xij', Lpq, tmp)
        vk = vk.reshape(dm.shape)
    return vj, vk
