from pyscf.ao2mo import incore
from pyscf.ao2mo.incore import iden_coeffs
from pyscfad import numpy as np
from pyscfad import lib
from pyscfad.ops import jit, vmap

def full(eri_ao, mo_coeff, verbose=0, compact=True, **kwargs):
    nao = mo_coeff.shape[0]
    if eri_ao.size != nao**4:
        raise NotImplementedError
    return general(eri_ao, (mo_coeff,)*4, verbose, compact)

def general(eri_ao, mo_coeffs, verbose=0, compact=True, **kwargs):
    nao = mo_coeffs[0].shape[0]
    if eri_ao.size == nao**4:
        return _eri_ao2mo_s1(eri_ao, mo_coeffs)
    else:
        return _eri_ao2mo_gen(eri_ao, mo_coeffs, verbose, compact)

@jit
def _eri_ao2mo_s1(eri_ao, mo_coeffs):
    nao = mo_coeffs[0].shape[0]
    #: np.einsum('pqrs,pi,qj,rk,sl->ijkl', eri_ao.reshape([nao]*4),
    #            mo_coeffs[0].conj(), mo_coeffs[1],
    #            mo_coeffs[2].conj(), mo_coeffs[3])
    eri_pqrl = np.dot(eri_ao.reshape(-1,nao), mo_coeffs[3])
    eri_lpqk = np.dot(eri_pqrl.T.reshape(-1,nao), mo_coeffs[2].conj())
    eri_klpj = np.dot(eri_lpqk.T.reshape(-1,nao), mo_coeffs[1])
    eri_jkli = np.dot(eri_klpj.T.reshape(-1,nao), mo_coeffs[0].conj())
    shape = [mo.shape[1] for mo in mo_coeffs]
    eri_ijkl = eri_jkli.T.reshape(shape)
    return eri_ijkl

def _eri_ao2mo_gen(eri_ao, mo_coeffs, verbose=0, compact=True):
    nao = mo_coeffs[0].shape[0]
    npair = nao*(nao+1)//2
    if eri_ao.shape == (npair,npair):
        return _eri_ao2mo_s4(eri_ao, mo_coeffs, compact)
    else:
        return incore.general(eri_ao, mo_coeffs, verbose=verbose, compact=compact)

def _eri_ao2mo_s4(eri_ao, mo_coeffs, compact=False):
    ni, nj, nk, nl = [mo.shape[1] for mo in mo_coeffs]
    iden_ij = iden_coeffs(mo_coeffs[0], mo_coeffs[1])
    iden_kl = iden_coeffs(mo_coeffs[2], mo_coeffs[3])

    def _half_e1(a, mo_k, mo_l):
        a = lib.unpack_tril(a, lib.SYMMETRIC)
        return np.einsum('uv,uk,vl->kl', a, mo_k, mo_l).flatten()

    eri_kl = vmap(_half_e1, in_axes=(0,None,None))(eri_ao, mo_coeffs[2], mo_coeffs[3])
    if iden_kl and compact:
        eri_kl = np.tril(eri_kl.reshape(-1,nk,nl))
    eri_ijkl = vmap(_half_e1, in_axes=(0,None,None))(eri_kl.T, mo_coeffs[0], mo_coeffs[1])
    if iden_ij and compact:
        eri_ijkl = np.tril(eri_ijkl.reshape(-1,ni,nj))
    eri_ijkl = eri_ijkl.T

    shape = eri_ijkl.shape
    if not compact or ((not iden_ij) and (not iden_kl)):
        shape = (ni, nj, nk, nl)
    elif not iden_ij:
        shape = (ni, nj, -1)
    elif not iden_kl:
        shape = (-1, nk, nl)
    eri_ijkl = eri_ijkl.reshape(shape)
    return eri_ijkl
