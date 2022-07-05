'''
EOM-EE-CCSD energies by diagonalizing CC Jacobian
Reference energy:
[0.46275482 0.73489818 1.09599866 1.15178554 1.60800261 1.70693685
 2.09393271 2.50583317 2.56362156]
'''
import numpy
import jax
from pyscfad import gto, scf, cc

mol = gto.Mole()
mol.atom = 'H 0. 0. 0.; H 0. 0. 1.1'
mol.basis = '631g*'
mol.verbose = 5
mol.incore_anyway = True
mol.build()

mf = scf.RHF(mol)
mf.kernel()
mycc = cc.RCCSD(mf)
mycc.kernel()

def amplitude_equation(mycc, vec, eris):
    t1, t2 = mycc.vector_to_amplitudes(vec)
    e1, e2 = mycc.amplitude_equation(t1, t2, eris)
    e = mycc.amplitudes_to_vector(e1, e2)
    return e

vec = mycc.amplitudes_to_vector(mycc.t1, mycc.t2)
eris = mycc.ao2mo(mycc.mo_coeff)
grad = jax.jacfwd(amplitude_equation, 1)(mycc, vec, eris)

w, x = numpy.linalg.eig(grad)
print("EOM-EE-CCSD singlet state energy:")
print(numpy.sort(w))
