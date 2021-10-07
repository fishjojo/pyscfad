from pyscfad.dft import rks
from pyscfad.dft import uks

def KS(mol, xc='LDA,VWN'):
    if mol.spin == 0:
        return RKS(mol, xc)
    else:
        return UKS(mol, xc)

KS.__doc__ = '''
A wrap function to create DFT object (RKS or UKS).\n
''' + rks.RKS.__doc__
DFT = KS

def RKS(mol, xc='LDA,VWN'):
    if mol.nelectron == 1:
        return uks.UKS(mol)

    assert (not mol.symmetry) or (mol.groupname == 'C1')
    assert (mol.spin == 0)

    return rks.RKS(mol, xc)

RKS.__doc__ = rks.RKS.__doc__

def UKS(mol, xc='LDA,VWN'):
    assert (not mol.symmetry) or (mol.groupname == 'C1')
    assert (mol.spin == 0)
    
    return uks.UKS(mol, xc)

UKS.__doc__ = uks.UKS.__doc__