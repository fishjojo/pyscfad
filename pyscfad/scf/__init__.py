from pyscfad.scf import hf
from pyscfad.scf import uhf
from pyscfad.scf import rohf

def RHF(mol, **kwargs):
    return hf.RHF(mol, **kwargs)

def UHF(mol, **kwargs):
    return uhf.UHF(mol, **kwargs)

def ROHF(mol, **kwargs):
    return rohf.ROHF(mol, **kwargs)
