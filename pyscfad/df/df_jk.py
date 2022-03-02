from jax import numpy as jnp

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
