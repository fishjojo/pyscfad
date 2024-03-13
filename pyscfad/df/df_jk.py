from pyscfad import numpy as np
from pyscfad import config
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
