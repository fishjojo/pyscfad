from pyscfad.pbc.dft import rks
from pyscfad.pbc.dft import krks

def RKS(cell, *args, **kwargs):
    return rks.RKS(cell, *args, **kwargs)

def KRKS(cell, *args, **kwargs):
    return krks.KRKS(cell, *args, **kwargs)
