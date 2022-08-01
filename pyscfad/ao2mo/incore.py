from pyscf import numpy as np
from pyscfad.lib import jit

def full(eri_ao, mo_coeff, verbose=0, compact=True, **kwargs):
    nao = mo_coeff.shape[0]
    if eri_ao.size != nao**4:
        raise NotImplementedError
    return general(eri_ao, (mo_coeff,)*4, verbose, compact)

@jit
def general(eri_ao, mo_coeffs, verbose=0, compact=True, **kwargs):
    nao = mo_coeffs[0].shape[0]
    #return np.einsum('pqrs,pi,qj,rk,sl->ijkl', eri_ao.reshape([nao]*4),
    #                  mo_coeffs[0].conj(), mo_coeffs[1],
    #                  mo_coeffs[2].conj(), mo_coeffs[3])
    eri_pqrl = np.dot(eri_ao.reshape(-1,nao), mo_coeffs[3])
    eri_lpqk = np.dot(eri_pqrl.T.reshape(-1,nao), mo_coeffs[2].conj())
    eri_klpj = np.dot(eri_lpqk.T.reshape(-1,nao), mo_coeffs[1])
    eri_jkli = np.dot(eri_klpj.T.reshape(-1,nao), mo_coeffs[0].conj())
    eri_ijkl = eri_jkli.T.reshape([nao]*4)
    return eri_ijkl
