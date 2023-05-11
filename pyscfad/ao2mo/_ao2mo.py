from pyscf import numpy as np

def nr_e2(eri, mo_coeff, orbs_slice, aosym='s1', mosym='s1', out=None,
          ao_loc=None):
    nrow = eri.shape[0]
    nao = mo_coeff.shape[0]
    k0, k1, l0, l1 = orbs_slice
    orb_k = mo_coeff[:,k0:k1]
    orb_l = mo_coeff[:,l0:l1]
    if aosym in ('s4', 's2', 's2kl'):
        from pyscfad.df import addons as df_addons
        eri = df_addons.restore('s1', eri, nao)
    else:
        eri = eri.reshape(-1,nao,nao)
    out = np.einsum('lpq,pi,qj->lij', eri, orb_k, orb_l).reshape(nrow,-1)
    return out
