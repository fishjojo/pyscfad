from pyscfad.scf import hf

def RHF(mol, **kwargs):
    return hf.RHF(mol, **kwargs)

def UHF(mol, **kwargs):
    return hf.RHF(mol, **kwargs)