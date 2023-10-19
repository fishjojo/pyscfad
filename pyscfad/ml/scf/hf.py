from functools import wraps
import numpy
from pyscf.lib import logger, module_method
from pyscf.data import nist
from pyscf.scf import hf as pyscf_hf
from pyscfad import ops
from pyscfad import numpy as np
from pyscfad.scf import hf

def cholesky_orth(s):
    L = np.linalg.cholesky(s)
    x = np.linalg.inv(L).T.conj()
    return x

class SCF(hf.SCF):
    def _eigh(self, h, s):
        if s is None:
            return np.linalg.eigh(h)

        # orthogonalize basis
        x = cholesky_orth(s)
        h_orth = np.dot(np.dot(x.T.conj(), h), x)
        e, c = np.linalg.eigh(h_orth)
        c = np.dot(x, c)
        return e, c

    def get_occ(self, mo_energy=None, mo_coeff=None):
        if mo_energy is None:
            mo_energy = self.mo_energy
        mo_energy = ops.convert_to_numpy(mo_energy)
        return super().get_occ(mo_energy)
        

if __name__ == '__main__':
    import torch
    from pyscf import gto

    mol = gto.Mole()
    mol.atom = 'H 0 0 0; F 0 0 0.9'
    mol.basis = 'sto3g'
    mol.build()

    fock = torch.rand(mol.nao, mol.nao, dtype=float)
    fock = .5 * (fock + fock.T.conj())
    fock = torch.autograd.Variable(fock, requires_grad=True)

    mf = SCF(mol)
    s = mf.get_ovlp()
    mo_energy, mo_coeff = mf.eig(fock, s)
    mo_occ = mf.get_occ(mo_energy) # get_occ returns a numpy array
    mo_occ = ops.convert_to_tensor(mo_occ)
    dm1 = mf.make_rdm1(mo_coeff, mo_occ)
    dip = mf.dip_moment(dm=dm1)
    dip_norm = np.linalg.norm(dip)
    dip_norm.backward()
    print(fock.grad)
