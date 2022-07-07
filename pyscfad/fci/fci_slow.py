from pyscf.fci import fci_slow as pyscf_fci_slow

kernel = pyscf_fci_slow.kernel

if __name__ == '__main__':
    from functools import reduce
    from pyscfad.lib import numpy as np
    from pyscfad import gto, ao2mo, scf

    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = [
        ['H', ( 1.,-1.    , 0.   )],
        ['H', ( 0.,-1.    ,-1.   )],
        ['H', ( 1.,-0.5   ,-1.   )],
        ['H', ( 0.,-0.    ,-1.   )],
        ['H', ( 1.,-0.5   , 0.   )],
        ['H', ( 0., 1.    , 1.   )],
    ]
    mol.basis = 'sto-3g'
    mol.build()

    m = scf.RHF(mol)
    m.kernel()
    norb = m.mo_coeff.shape[1]
    nelec = mol.nelectron - 2
    h1e = reduce(np.dot, (m.mo_coeff.T, m.get_hcore(), m.mo_coeff))
    eri = ao2mo.incore.full(m._eri, m.mo_coeff, compact=False)
    eri = eri.reshape(norb,norb,norb,norb)

    e1 = kernel(h1e, eri, norb, nelec)[0]
    print(e1, e1 - -7.9766331504361414)
