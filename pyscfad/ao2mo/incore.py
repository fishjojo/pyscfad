from pyscfad.lib import numpy as np

def full(eri_ao, mo_coeff, verbose=0, compact=True, **kwargs):
    return general(eri_ao, (mo_coeff,)*4, verbose, compact)

def general(eri_ao, mo_coeffs, verbose=0, compact=True, **kwargs):
    nao = mo_coeffs[0].shape[0]

    if eri_ao.size == nao**4:
        return np.einsum('pqrs,pi,qj,rk,sl->ijkl', eri_ao.reshape([nao]*4),
                          mo_coeffs[0].conj(), mo_coeffs[1],
                          mo_coeffs[2].conj(), mo_coeffs[3])
    else:
        raise NotImplementedError
