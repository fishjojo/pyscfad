from functools import partial
import numpy
from jax import custom_vjp
from pyscf.gto.moleintor import _get_intor_and_comp
from pyscf.pbc.gto import Cell
from pyscfad import numpy as np
from pyscfad.ops import custom_jvp
from pyscfad.gto._moleintor_jvp import _int1e_fill_jvp_r0_s2

@partial(custom_jvp, nondiff_argnums=tuple(range(1,7)))
def _pbc_intor(mol, intor, comp=None, hermi=0, kpts=None, kpt=None,
               shls_slice=None):
    return Cell.pbc_intor(mol.view(Cell), intor, comp, hermi, kpts, kpt, shls_slice)

@_pbc_intor.defjvp
def _pbc_intor_jvp(intor, comp, hermi, kpts, kpt, shls_slice,
                   primals, tangents):
    if hermi != 1:
        raise NotImplementedError
    if shls_slice is not None:
        raise NotImplementedError

    mol, = primals
    mol_t, = tangents

    primal_out = _pbc_intor(mol, intor, comp, hermi, kpts, kpt, shls_slice)
    tangent_out = None
    if mol.coords is not None:
        if intor.startswith("int1e"):
            intor_ip = intor.replace("int1e_", "int1e_ip")
        else:
            raise NotImplementedError
        tangent_out = _int1e_jvp_r0(mol, mol_t, intor_ip, hermi, kpts, kpt, shls_slice)

    if mol.ctr_coeff is not None:
        pass
    if mol.exp is not None:
        pass

    return primal_out, tangent_out

def _int1e_jvp_r0(mol, mol_t, intor, hermi, kpts, kpt, shls_slice):
    _, comp = _get_intor_and_comp(intor)
    if comp != 3:
        raise NotImplementedError

    s1 = _pbc_intor(mol, intor, comp=comp, hermi=0,
                    kpts=kpts, kpt=kpt, shls_slice=shls_slice)

    gamma = False
    if getattr(s1, "ndim", None) == 2:
        s1 = [s1,]
        gamma = True

    aoslices = mol.aoslice_by_atom()[:,2:4]
    idx = np.arange(mol.nao)[None,:,None]
    tangent_out = [_int1e_fill_jvp_r0_s2(-s1_k, mol_t.coords, aoslices, idx) for s1_k in s1]

    if gamma:
        tangent_out = tangent_out[0]
    return tangent_out

@partial(custom_vjp, nondiff_argnums=tuple(range(1,7)))
def _pbc_intor_rev(mol, intor, comp=None, hermi=0, kpts=None, kpt=None,
                   shls_slice=None):
    return Cell.pbc_intor(mol.view(Cell), intor, comp, hermi, kpts, kpt, shls_slice)

def _pbc_intor_fwd(mol, intor, comp, hermi, kpts, kpt, shls_slice):
    primal_out = _pbc_intor_rev(mol, intor, comp, hermi, kpts, kpt, shls_slice)
    res = (mol,)
    return primal_out, res

def _pbc_intor_bwd(intor, comp, hermi, kpts, kpt, shls_slice, res, y_bar):
    mol, = res

    if mol.coords is not None:
        if intor.startswith("int1e"):
            intor_ip = intor.replace("int1e_", "int1e_ip")
        else:
            raise NotImplementedError
    partial_r0 = _int1e_partial_r0(mol, intor_ip, kpts, kpt, shls_slice)
    if isinstance(partial_r0, numpy.ndarray):
        r0_bar = np.einsum("nxij,ij->nx", partial_r0, y_bar)
    else:
        r0_bar = []
        for item in partial_r0:
            r0_bar.append(np.einsum("nxij,ij->nx", item, y_bar))

    ctr_coeff_bar=None
    if mol.ctr_coeff is not None:
        pass

    exp_bar=None
    if mol.exp is not None:
        pass

    mol.coords = r0_bar
    return (mol,)

def _int1e_partial_r0(mol, intor, kpts, kpt, shls_slice):
    aoslices = mol.aoslice_by_atom()
    nao = mol.nao
    s1 = Cell.pbc_intor(mol.view(Cell), intor, comp=3, hermi=0,
                        kpts=kpts, kpt=kpt, shls_slice=shls_slice)

    if isinstance(s1, numpy.ndarray):
        s1 = [s1,]
    nkpts = len(s1)

    partial_r0 = []
    for k in range(nkpts):
        s1_k = s1[k]
        grad = numpy.zeros((mol.natm,3,nao,nao), dtype=s1_k.dtype)
        for ia in range(mol.natm):
            p0, p1 = aoslices [ia,2:]
            grad[ia,:,p0:p1] += -s1_k[:,p0:p1]
            grad[ia,:,:,p0:p1] += -s1_k[:,p0:p1].transpose(0,2,1).conj()
        #grad += grad.transpose(0,1,3,2)
        partial_r0.append(grad)
    if nkpts == 1:
        partial_r0 = partial_r0[0]
    return partial_r0

_pbc_intor_rev.defvjp(_pbc_intor_fwd, _pbc_intor_bwd)
