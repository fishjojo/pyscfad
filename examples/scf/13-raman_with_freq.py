import jax
from pyscfad import gto
from pyscfad import scf
from pyscfad.prop.polarizability import rhf

mol = gto.Mole()
mol.atom = '''h  ,  0.   0.   0.
              F  ,  0.   0.   .917'''
mol.basis = '631g'
mol.verbose = 5
mol.build(trace_ctr_coeff=False, trace_exp=False)

def polar(mol, freq=0.0):
    mf = scf.RHF(mol)
    mf.kernel()
    alpha = rhf.Polarizability(mf).polarizability_with_freq(freq=freq)
    return alpha

chi = jax.jacfwd(polar)(mol, freq=0.1).coords
print("Raman tensor")
print(chi)
