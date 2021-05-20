from functools import partial
import numpy
from jax import custom_jvp, custom_vjp
from pyscf.pbc.gto import cell
from pyscfad.lib import numpy as jnp

@partial(custom_jvp, nondiff_argnums=tuple(range(1,7)))
def _pbc_intor(mol, intor, comp=None, hermi=0, kpts=None, kpt=None,
               shls_slice=None):
    return cell.Cell.pbc_intor(mol, intor, comp, hermi, kpts, kpt, shls_slice)

@_pbc_intor.defjvp
def _pbc_intor_jvp(intor, comp, hermi, kpts, kpt, shls_slice,
                   primals, tangents):
    mol, = primals
    mol_t, = tangents

    primal_out = _pbc_intor(mol, intor, comp, hermi, kpts, kpt, shls_slice)
    tangent_out = None
    if mol.coords is not None:
        if intor.startswith("int1e"):
            intor_ip = intor.replace("int1e_", "int1e_ip")
        else:
            raise NotImplementedError
        tangent_out = _int1e_jvp_r0(mol, mol_t, intor_ip, kpts, kpt, shls_slice)

    if mol.ctr_coeff is not None:
        pass
    if mol.exp is not None:
        pass

    return primal_out, tangent_out

def _int1e_jvp_r0(mol, mol_t, intor, kpts, kpt, shls_slice):
    coords_t = mol_t.coords
    aoslices = mol.aoslice_by_atom()
    nao = mol.nao
    s1 = cell.Cell.pbc_intor(mol, intor, comp=3, hermi=0, 
                             kpts=kpts, kpt=kpt, shls_slice=shls_slice)

    gamma = False
    if isinstance(s1, numpy.ndarray):
        s1 = [s1,]
        gamma = True

    def get_tangent_out(s1_k):
        grad = numpy.zeros((mol.natm,3,nao,nao), dtype=s1_k.dtype)
        for ia in range(mol.natm):
            p0, p1 = aoslices [ia,2:]
            grad[ia,:,p0:p1] += -s1_k[:,p0:p1]
        tangent_out_k = jnp.einsum("nxij,nx->ij", grad, coords_t)
        tangent_out_k += tangent_out_k.T.conj()
        return tangent_out_k

    # FIXME this may be slow with many k-points
    tangent_out = [get_tangent_out(s1_k) for s1_k in s1]

    if gamma:
        tangent_out = tangent_out[0]
    return tangent_out


@partial(custom_vjp, nondiff_argnums=tuple(range(1,7)))
def _pbc_intor_rev(mol, intor, comp=None, hermi=0, kpts=None, kpt=None,
                   shls_slice=None):
    return cell.Cell.pbc_intor(mol, intor, comp, hermi, kpts, kpt, shls_slice)

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
        r0_bar = jnp.einsum("nxij,ij->nx", partial_r0, y_bar)
    else:
        r0_bar = []
        for item in partial_r0:
            r0_bar.append(jnp.einsum("nxij,ij->nx", item, y_bar))

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
    s1 = cell.Cell.pbc_intor(mol, intor, comp=3, hermi=0,
                             kpts=kpts, kpt=kpt, shls_slice=shls_slice)

    if isinstance(s1, numpy.ndarray):
        s1 = [s1,]
    nkpts = len(s1)

    partial = []
    for k in range(nkpts):
        s1_k = s1[k]
        grad = numpy.zeros((mol.natm,3,nao,nao), dtype=s1_k.dtype)
        for ia in range(mol.natm):
            p0, p1 = aoslices [ia,2:]
            grad[ia,:,p0:p1] += -s1_k[:,p0:p1]
            grad[ia,:,:,p0:p1] += -s1_k[:,p0:p1].transpose(0,2,1).conj()
        #grad += grad.transpose(0,1,3,2)
        partial.append(grad)
    if nkpts == 1:
        partial = partial[0]
    return partial

_pbc_intor_rev.defvjp(_pbc_intor_fwd, _pbc_intor_bwd)
