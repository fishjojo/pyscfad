from pyscf.df import addons as pyscf_addons
from pyscfad import numpy as np
from pyscfad import lib
from pyscfad import ao2mo
from pyscfad.gto._mole_helper import setup_exp, setup_ctr_coeff

class load(ao2mo.load):
    def __init__(self, eri, dataname='j3c'):
        ao2mo.load.__init__(self, eri, dataname)

def make_auxmol(mol, auxbasis=None):
    auxmol = pyscf_addons.make_auxmol(mol, auxbasis=auxbasis)
    if mol.exp is not None:
        auxmol.exp = np.asarray(setup_exp(auxmol)[0])
    if mol.ctr_coeff is not None:
        auxmol.ctr_coeff = np.asarray(setup_ctr_coeff(auxmol)[0])
    return auxmol

def restore(symmetry, cderi, nao):
    '''convert the three center integral between different
    permutation symmetries
    '''
    npair = nao*(nao+1)//2
    if symmetry in ('1', 's1'):
        if cderi.ndim == 3 and cderi.shape[-1] == nao:
            return cderi
        elif cderi.ndim == 2 and cderi.shape[-1] == nao**2:
            return cderi.reshape(-1,nao,nao)
        elif cderi.ndim == 2 and cderi.shape[-1] == npair:
            return lib.unpack_tril(cderi, lib.SYMMETRIC)
        else:
            raise RuntimeError(f'cderi shape {cderi.shape} incompatible with nao {nao}')
    elif symmetry in ('2', 's2', 's2ij'):
        if cderi.ndim == 2 and cderi.shape[-1] == npair:
            return cderi
        elif cderi.ndim == 3 and cderi.shape[-1] == nao:
            return lib.pack_tril(cderi)
        elif cderi.ndim == 2 and cderi.shape[-1] == nao**2:
            return lib.pack_tril(cderi.reshape(-1,nao,nao))
        else:
            raise RuntimeError(f'cderi shape {cderi.shape} incompatible with nao {nao}')
    else:
        raise KeyError(f'Unsupported symmetry {symmetry}.')
