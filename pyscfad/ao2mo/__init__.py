from pyscfad.lib import isarray
from pyscfad.ao2mo import incore

def general(eri_or_mol, mo_coeffs, erifile=None, dataname='eri_mo', intor='int2e',
            *args, **kwargs):
    if isarray(eri_or_mol):
        return incore.general(eri_or_mol, mo_coeffs, *args, **kwargs)
    else:
        raise NotImplementedError
