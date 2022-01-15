from pyscf.df import addons as pyscf_addons
from pyscfad.gto._mole_helper import setup_exp, setup_ctr_coeff

def make_auxmol(mol, auxbasis=None):
    auxmol = pyscf_addons.make_auxmol(mol, auxbasis=auxbasis)
    if mol.exp is not None:
        auxmol.exp, _, _ = setup_exp(auxmol)
    if mol.ctr_coeff is not None:
        auxmol.ctr_coeff, _, _ = setup_ctr_coeff(auxmol)
    return auxmol
