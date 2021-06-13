from pyscfad.pbc.dft import rks

def RKS(cell, *args, **kwargs):
    return rks.RKS(cell, *args, **kwargs)
