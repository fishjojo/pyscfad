import copy
import itertools
import numpy as np
from pyscf.data.nist import HARTREE2EV
from pyscfad import lib
from pyscfad.lib import numpy as jnp
from pyscfad.dft import rks

def make_minao_lo(ks, minao_ref='minao'):
    from pyscf import lo
    mol = ks.mol
    nao = mol.nao_nr()
    ovlp = ks.get_ovlp()
    C_ao_minao, labels = proj_ref_ao(mol, minao=minao_ref,
                                     return_labels=True)
    C_ao_minao = lo.vec_lowdin(C_ao_minao, ovlp)
    labels = np.asarray(labels)

    C_ao_lo = np.zeros((nao, nao))
    for idx, lab in zip(ks.U_idx, ks.U_lab):
        idx_minao = [i for i, l in enumerate(labels) if l in lab]
        assert len(idx_minao) == len(idx)
        C_ao_sub = C_ao_minao[..., idx_minao]
        C_ao_lo[..., idx] = C_ao_sub
    return C_ao_lo

def proj_ref_ao(mol, minao='minao', return_labels=False):
    from pyscf.lo import iao
    from pyscf.gto import mole
    import scipy.linalg as la

    pmol = iao.reference_mol(mol, minao)
    s1 = np.asarray(mol.intor('int1e_ovlp', hermi=1))
    s2 = np.asarray(pmol.intor('int1e_ovlp', hermi=1))
    s12 = np.asarray(mole.intor_cross('int1e_ovlp', mol, pmol))
    C_ao_lo = np.zeros((s1.shape[-1], s2.shape[-1]))
    s1cd = la.cho_factor(s1)
    C_ao_lo = la.cho_solve(s1cd, s12)
    if return_labels:
        labels = pmol.ao_labels()
        return C_ao_lo, labels
    else:
        return C_ao_lo



def set_U(mol, U_idx, U_val):
    assert len(U_idx) == len(U_val)
    _U_val = []
    _U_idx = []
    _U_lab = []

    lo_labels = np.asarray(mol.ao_labels())
    for i, idx in enumerate(U_idx):
        if isinstance(idx, str):
            lab_idx = mol.search_ao_label(idx)
            labs = lo_labels[lab_idx]
            labs = zip(lab_idx, labs)
            for j, idxj in itertools.groupby(labs, key=lambda x: x[1].split()[0]):
                _U_idx.append(list(list(zip(*idxj))[0]))
                _U_val.append(U_val[i])
        else:
            _U_idx.append(copy.deepcopy(idx))
            _U_val.append(U_val[i])
    _U_val = jnp.asarray(_U_val) / HARTREE2EV
    for idx, val in zip(_U_idx, _U_val):
        _U_lab.append(lo_labels[idx])
    return _U_val, _U_idx, _U_lab


def get_veff(ks, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
    vxc = rks.get_veff(ks, mol, dm, dm_last, vhf_last, hermi)

    C_ao_lo = ks.C_ao_lo
    ovlp = ks.get_ovlp()
    nlo = C_ao_lo.shape[-1]

    C_inv = jnp.dot(C_ao_lo.conj().T, ovlp)
    rdm1_lo = jnp.dot(jnp.dot(C_inv, dm), C_inv.conj().T)

    E_U = 0.0
    for idx, val, lab in zip(ks.U_idx, ks.U_val, ks.U_lab):
        lab_string = " "
        for l in lab:
            lab_string += "%9s" %(l.split()[-1])
        lab_sp = lab[0].split()
        U_mesh = jnp.ix_(idx, idx)

        C_k = C_ao_lo[:, idx]
        P_k = rdm1_lo[U_mesh]
        SC = jnp.dot(ovlp, C_k)
        vxc.vxc += jnp.dot(jnp.dot(SC, (jnp.eye(P_k.shape[-1]) - P_k)
                               * (val * 0.5)), SC.conj().T)
        E_U += (val * 0.5) * (P_k.trace() - jnp.dot(P_k, P_k).trace() * 0.5)

    vxc.exc += E_U
    return vxc

@lib.dataclass
class RKSpU(rks.RKS):
    U_val: jnp.array = lib.field(pytree_node=True, default=None)
    U_idx: list = lib.field(default_factory=list)
    U_lab: list = lib.field(default_factory=list)
    C_ao_lo: jnp.array = None

    def __post_init__(self):
        rks.RKS.__post_init__(self)

    get_veff = get_veff

if __name__ == "__main__":
    import jax
    from pyscfad import gto
    mol = gto.Mole()
    mol.atom = 'O 0. 0. 0.; H 0. , -0.757 , 0.587; H 0. , 0.757 , 0.587'
    mol.basis = '631g'
    mol.build()

    U_idx = ["0 O 2p"]
    U_val = [5.0]
    mf = RKSpU(mol)
    mf.U_val, mf.U_idx, mf.U_lab = set_U(mol, U_idx, U_val)
    mf.C_ao_lo = make_minao_lo(mf)
    dm0 = mf.get_init_guess()
    jac = jax.jacrev(mf.__class__.kernel)(mf, dm0=dm0)
    print(jac.U_val)
    print(jac.mol.coords)
