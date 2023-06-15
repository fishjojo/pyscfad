from pyscfad.ao2mo import incore
from pyscfad.ao2mo.addons import load, restore

def general(eri_or_mol, mo_coeffs, *args,
            erifile=None, dataname='eri_mo', intor='int2e',
            **kwargs):
    from pyscfad import lib
    if lib.isarray(eri_or_mol):
        return incore.general(eri_or_mol, mo_coeffs, *args, **kwargs)
    else:
        raise NotImplementedError
