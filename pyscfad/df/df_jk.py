from pyscf import numpy as np
from .addons import restore

def get_jk(dfobj, dm, hermi=1, with_j=True, with_k=True, direct_scf_tol=1e-13):
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
