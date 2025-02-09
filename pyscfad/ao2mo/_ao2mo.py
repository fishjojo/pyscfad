from pyscfad import numpy as np
from pyscfad import config
from ._ao2mo_opt import nr_e2 as nr_e2_opt

def nr_e2(eri, mo_coeff, orbs_slice, aosym='s1', mosym='s1', out=None,
          ao_loc=None):
    if config.moleintor_opt:
        return nr_e2_opt(eri, mo_coeff, orbs_slice,
                         aosym=aosym, mosym=mosym,
                         out=out, ao_loc=ao_loc)
    else:
        return nr_e2_gen(eri, mo_coeff, orbs_slice, aosym=aosym)

def nr_e2_gen(eri, mo_coeff, orbs_slice, aosym='s1'):
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
