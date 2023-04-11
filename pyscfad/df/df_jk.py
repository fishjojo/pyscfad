from pyscf import numpy as np
from pyscfad import lib
from pyscfad.lib import vmap

def get_jk(dfobj, dm, hermi=1, with_j=True, with_k=True, direct_scf_tol=1e-13):
    nao = dfobj.mol.nao
    dms = dm.reshape(-1, nao, nao)

    Lpq = dfobj._cderi
    if Lpq.shape[-1] == nao**2:
        Lpq = Lpq.reshape(-1,nao,nao)
    else:
        Lpq = vmap(lib.unpack_tril, in_axes=(0,None))(Lpq, lib.SYMMETRIC)

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
