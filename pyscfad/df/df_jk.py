from pyscfad.lib import numpy as np

def get_jk(dfobj, dm, hermi=1, with_j=True, with_k=True, direct_scf_tol=1e-13):
    assert(with_j or with_k)

    nao = dfobj.mol.nao
    dms = dm.reshape(-1, nao, nao)

    vj = vk = 0
    Lpq = dfobj._cderi
    if with_j:
        tmp = np.einsum("Lpq,xpq->xL", Lpq, dms)
        vj = np.einsum("Lpq,xL->xpq", Lpq, tmp)
        vj = vj.reshape(dm.shape)

    if with_k:
        tmp = np.einsum('Lij,xjk->xLki', Lpq, dms)
        vk = np.einsum('Lki,xLkj->xij', Lpq, tmp)
        vk = vk.reshape(dm.shape)
    return vj, vk
