from pyscf.ao2mo import addons as pyscf_addons
from pyscfad import lib
from pyscfad.ops import vmap

class load(pyscf_addons.load):
    def __enter__(self):
        if lib.isarray(self.eri):
            return self.eri
        else:
            raise NotImplementedError


def restore(symmetry, eri, norb, tao=None):
    targetsym = _stand_sym_code(symmetry)
    if targetsym not in ('8', '4', '1', '2kl', '2ij'):
        raise ValueError(f'symmetry = {symmetry}')

    npair = norb*(norb+1)//2
    if eri.size == npair*npair:
        if targetsym == '1':
            return _convert_s4_to_s1(eri, norb)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


def _convert_s4_to_s1(eri, norb):
    npair = norb*(norb+1)//2
    eri = eri.reshape(npair,npair)
    eri = vmap(lib.unpack_tril, (0,None))(eri, lib.SYMMETRIC)
    eri = eri.transpose(1,2,0).reshape(norb*norb, -1)
    eri = vmap(lib.unpack_tril, (0,None))(eri, lib.SYMMETRIC)
    eri = eri.transpose(1,2,0).reshape([norb,]*4)
    return eri


def _stand_sym_code(sym):
    if isinstance(sym, int):
        return str(sym)
    elif 's' == sym[0]:
        return sym[1:]
    else:
        return sym
