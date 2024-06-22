from pyscfad.ao2mo import incore
from pyscfad.ao2mo.addons import load, restore

def general(eri_or_mol, mo_coeffs, *args,
            erifile=None, dataname='eri_mo', intor='int2e',
            **kwargs):
    if hasattr(eri_or_mol, 'shape'):
        return incore.general(eri_or_mol, mo_coeffs, *args, **kwargs)
    else:
        raise NotImplementedError
