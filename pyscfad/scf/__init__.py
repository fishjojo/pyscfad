from pyscfad.scf import hf
rhf = hf
from pyscfad.scf import uhf

def HF(mol, *args):
    if mol.nelectron == 1 or mol.spin == 0:
        return RHF(mol, *args)
    else:
        return UHF(mol, *args)

def RHF(mol, **kwargs):
    return hf.RHF(mol, **kwargs)

def UHF(mol, **kwargs):
    if mol.nelectron == 1:
        return uhf.HF1e(mol, *args)
    else:
        return uhf.UHF(mol, *args)