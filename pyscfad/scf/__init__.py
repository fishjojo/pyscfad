from pyscfad.scf import hf
from pyscfad.scf import uhf

def RHF(mol, **kwargs):
    return hf.RHF(mol, **kwargs)

def UHF(mol, **kwargs):
    return uhf.UHF(mol, **kwargs)
