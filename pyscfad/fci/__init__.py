from functools import reduce
from pyscf import numpy as np
#from pyscfad.lib import numpy as np
from pyscfad import ao2mo
from pyscfad.fci import fci_slow
from pyscfad.fci.fci_slow import fci_ovlp

def solve_fci(mf, nroots=1):
    mol = mf.mol
    mo_coeff = mf.mo_coeff
    norb = mo_coeff.shape[-1]
    nelec = mol.nelectron
    h1e = reduce(np.dot, (mo_coeff.T, mf.get_hcore(), mo_coeff))
    eri = ao2mo.incore.full(mf._eri, mo_coeff, compact=False)
    eri = eri.reshape(norb,norb,norb,norb)
    e, fcivec = fci_slow.kernel(h1e, eri, norb, nelec, mf.energy_nuc(), nroots=nroots)
    return e, fcivec
