from pyscfad.pbc.scf import hf
from pyscfad.pbc.scf import khf

def RHF(cell, **kwargs):
    return hf.RHF(cell, **kwargs)

def KRHF(cell, **kwargs):
    return khf.KRHF(cell, **kwargs)
