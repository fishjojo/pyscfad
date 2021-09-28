from pyscfad.scf import hf
from pyscfad.scf import uhf

def RHF(mol, **kwargs):
    return hf.RHF(mol, **kwargs)

def UHF(mol, **kwargs):
    if mol.nelectron == 1:
        return uhf.HF1e(mol, **kwargs)
    else:
        return uhf.UHF(mol, **kwargs)
