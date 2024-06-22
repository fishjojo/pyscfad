from functools import partial
import numpy

from pyscf import ao2mo
from pyscf.gto import mole as pyscf_mole
from pyscf.gto import ATOM_OF
from pyscf.gto.moleintor import _get_intor_and_comp

from pyscfad import numpy as np
from pyscfad import ops
from pyscfad.ops import (
    custom_jvp,
    jit,
    vmap,
)
from ._mole_helper import (
    setup_exp,
    setup_ctr_coeff,
    get_fakemol_exp,
    get_fakemol_cs,
)
from ._moleintor_helper import (
    int1e_get_dr_order,
    int2e_get_dr_order,
    int1e_dr1_name,
    int2e_dr1_name,
    _intor_impl,
    _intor_cross_impl,
    get_bas_label,
    promote_xyz,
)
from pyscfad.gto import _pyscf_moleintor as moleintor

SET_RC = ['rinv',]
_S_NORM = 0.282094791773878143 # normalization factor for s orbital
_P_NORM = 0.488602511902919921 # normalization factor for p orbital

@partial(custom_jvp, nondiff_argnums=(0,3,4))
def intor_cross(intor, mol1, mol2, comp=None, grids=None):
    return _intor_cross_impl(intor, mol1, mol2, comp=comp, grids=grids)

@intor_cross.defjvp
def intor_cross_jvp(intor, comp, grids,
                    primals, tangents):
    mol1, mol2 = primals
    mol1_t, mol2_t = tangents

    primal_out = intor_cross(intor, mol1, mol2, comp=comp, grids=grids)
    tangent_out = np.zeros_like(primal_out)

    nao1 = mol1.nao
    nao2 = mol2.nao
    if mol1.coords is not None:
        aoslices1 = mol1.aoslice_by_atom()[:,2:4]
        intor_ip_bra, _ = int1e_dr1_name(intor)
        s1a = -intor_cross(intor_ip_bra, mol1, mol2, comp=None, grids=grids).reshape(3,-1,nao1,nao2)

        idx1 = np.arange(nao1)
        grad1 = _gen_int1e_fill_grad_r0(s1a, aoslices1, idx1[None,None,:,None])
        tangent_out += np.einsum('nxyij,nx->yij', grad1, mol1_t.coords).reshape(primal_out.shape)

    if mol2.coords is not None:
        aoslices2 = mol2.aoslice_by_atom()[:,2:4]
        _, intor_ip_ket = int1e_dr1_name(intor)
        s1b = -intor_cross(intor_ip_ket, mol1, mol2, comp=None, grids=grids)

        order_a = int1e_get_dr_order(intor_ip_ket)[0]
        s1b = s1b.reshape(3**order_a,3,-1,nao1,nao2).transpose(1,0,2,3,4).reshape(3,-1,nao1,nao2)
        idx2 = np.arange(nao2)
        grad2 = _gen_int1e_fill_grad_r0(s1b, aoslices2, idx2[None,None,None,:])
        tangent_out += np.einsum('nxyij,nx->yij', grad2, mol2_t.coords).reshape(primal_out.shape)

    if mol1.ctr_coeff is not None:
        pass
    if mol1.exp is not None:
        pass

    if mol2.ctr_coeff is not None:
        pass
    if mol2.exp is not None:
        pass

    return primal_out, tangent_out

def getints2c_rc(mol, intor, shls_slice=None, comp=None,
                 hermi=0, out=None, rc_deriv=None):
    if rc_deriv is None or not any(rc in intor for rc in SET_RC):
        return intor2c(mol, intor, shls_slice=shls_slice, comp=comp, hermi=hermi, out=out)
    else:
        return _getints2c_rc(mol, intor, shls_slice, comp, hermi, out, rc_deriv)

@partial(custom_jvp, nondiff_argnums=(1,2,3,4,5,6))
def _getints2c_rc(mol, intor, shls_slice=None, comp=None,
                  hermi=0, out=None, rc_deriv=None):
    return _intor_impl(mol, intor, comp=comp, hermi=hermi,
                       shls_slice=shls_slice, out=out)

@_getints2c_rc.defjvp
def _getints2c_rc_jvp(intor, shls_slice, comp, hermi, out, rc_deriv,
                      primals, tangents):
    if shls_slice is not None:
        raise NotImplementedError

    mol, = primals
    mol_t, = tangents
    primal_out = _getints2c_rc(mol, intor, shls_slice, comp, hermi, out)
    tangent_out = np.zeros_like(primal_out)

    if mol.coords is not None:
        intor_ip_bra, intor_ip_ket = int1e_dr1_name(intor)
        tangent_out += _gen_int1e_jvp_r0(mol, mol_t, intor_ip_bra, intor_ip_ket, rc_deriv)

    if mol.ctr_coeff is not None:
        tangent_out += _int1e_jvp_cs(mol, mol_t, intor, shls_slice, comp, hermi)

    if mol.exp is not None:
        tangent_out += _int1e_jvp_exp(mol, mol_t, intor, shls_slice, comp, hermi)
    return primal_out, tangent_out

@partial(custom_jvp, nondiff_argnums=tuple(range(1,8)))
def intor2c(mol, intor, comp=None, hermi=0, aosym='s1', out=None,
            shls_slice=None, grids=None):
    return _intor_impl(mol, intor, comp=comp, hermi=hermi, aosym=aosym, out=out,
                       shls_slice=shls_slice, grids=grids)

@intor2c.defjvp
def intor2c_jvp(intor, comp, hermi, aosym, out, shls_slice, grids,
                primals, tangents):
    if '_spinor' in intor:
        msg = 'Integrals for spinors are not differentiable.'
        raise NotImplementedError(msg)
    if grids is not None:
        msg = 'Integrals on grids are not differentiable.'
        raise NotImplementedError(msg)
    if hermi == 2:
        msg = 'Integrals with anti-hermitian symmetry are not differentiable.'
        raise NotImplementedError(msg)
    if shls_slice is not None:
        msg = f'AD for integral {intor} with subblocks of shells are not supported.'
        raise NotImplementedError(msg)

    mol, = primals
    mol_t, = tangents

    primal_out = intor2c(mol, intor, comp=comp, hermi=hermi, aosym=aosym, out=out,
                         shls_slice=shls_slice, grids=grids)

    tangent_out = np.zeros_like(primal_out)
    fname = intor.replace('_sph', '').replace('_cart', '')
    if mol.coords is not None:
        intor_ip_bra = intor_ip_ket = intor_ip = None
        if intor.startswith('ECPscalar'):
            intor_ip = intor.replace('ECPscalar', 'ECPscalar_ipnuc')
        elif fname == 'int1e_r':
            intor_ip = intor.replace('int1e_r', 'int1e_irp')
        elif fname.startswith('int1e') or fname.startswith('int2c2e'):
            intor_ip_bra, intor_ip_ket = int1e_dr1_name(intor)
        else:
            raise NotImplementedError(f'Integral {intor} is not supported for AD.')

        if intor_ip_bra or intor_ip_ket:
            tangent_out += _gen_int1e_jvp_r0(mol, mol_t,
                                intor_ip_bra, intor_ip_ket, hermi=hermi).reshape(tangent_out.shape)
            if 'nuc' in intor_ip_bra and 'nuc' in intor_ip_ket:
                intor_ip_bra = intor_ip_bra.replace('nuc', 'rinv')
                intor_ip_ket = intor_ip_ket.replace('nuc', 'rinv')
                tangent_out += _gen_int1e_nuc_jvp_rc(mol, mol_t,
                                intor_ip_bra, intor_ip_ket, hermi=hermi).reshape(tangent_out.shape)
        elif fname == 'int1e_r':
            tangent_out += _int1e_r_jvp_r0(mol, mol_t, intor_ip)
        else:
            tangent_out += _int1e_jvp_r0(mol, mol_t, intor_ip)

        intor_ip = None
        if intor.startswith('ECPscalar'):
            intor_ip = intor.replace('ECPscalar', 'ECPscalar_iprinv')
        if intor_ip:
            tangent_out += _int1e_nuc_jvp_rc(mol, mol_t, intor_ip)

    if mol.ctr_coeff is not None:
        tangent_out += _int1e_jvp_cs(mol, mol_t, intor, shls_slice, comp, hermi)

    if mol.exp is not None:
        tangent_out += _int1e_jvp_exp(mol, mol_t, intor, shls_slice, comp, hermi)
    return primal_out, tangent_out

def intor3c(mol, intor, comp=None, hermi=0, aosym='s1', out=None,
            shls_slice=None, grids=None):
    return _intor_impl(mol, intor, comp=comp, hermi=hermi, aosym=aosym, out=out,
                       shls_slice=shls_slice, grids=grids)

@partial(custom_jvp, nondiff_argnums=tuple(range(1,8)))
def intor4c(mol, intor, comp=None, hermi=0, aosym='s1', out=None,
            shls_slice=None, grids=None):
    if (shls_slice is None and aosym=='s1'
            and intor in ['int2e', 'int2e_sph', 'int2e_cart']):
        eri8 = _intor_impl(mol, intor, comp=comp, aosym='s8',
                           shls_slice=shls_slice, out=out)
        eri = ao2mo.restore(aosym, eri8, mol.nao)
        del eri8
    else:
        eri = _intor_impl(mol, intor, comp=comp, aosym=aosym,
                          shls_slice=shls_slice, out=out)
    return eri

@intor4c.defjvp
def intor4c_jvp(intor, comp, hermi, aosym, out, shls_slice, grids,
                primals, tangents):
    if '_spinor' in intor:
        msg = 'Integrals for spinors are not differentiable.'
        raise NotImplementedError(msg)
    if grids is not None:
        msg = 'Integrals on grids are not differentiable.'
        raise NotImplementedError(msg)
    if shls_slice is not None:
        msg = f'AD for integral {intor} with subblocks of shells are not supported.'
        raise NotImplementedError(msg)
    if aosym != 's1':
        msg = f'AD for integral {intor} with aosym = {aosym} is not supported.'
        raise NotImplementedError(msg)

    mol, = primals
    primal_out = intor4c(mol, intor, comp=comp, hermi=hermi, aosym=aosym, out=out,
                         shls_slice=shls_slice, grids=grids)

    mol_t, = tangents
    tangent_out = np.zeros_like(primal_out)

    fname = intor.replace('_sph', '').replace('_cart', '')
    if mol.coords is not None:
        intor1, intor2, intor3, intor4 = int2e_dr1_name(intor)
        if fname[-6:-4] == 'dr':
            orders = int2e_get_dr_order(intor)
            if orders[0] == 0 and orders[1] == 0:
                intor2 = None
            if orders[2] == 0 and orders[3] == 0:
                intor4 = None
            tangent_out += _gen_int2e_jvp_r0(mol, mol_t, [intor1, intor2, intor3, intor4])
        else:
            tangent_out += _int2e_jvp_r0(mol, mol_t, intor1)

    if mol.ctr_coeff is not None:
        tangent_out += _int2e_jvp_cs(mol, mol_t, intor, shls_slice, comp)
    if mol.exp is not None:
        tangent_out += _int2e_jvp_exp(mol, mol_t, intor, shls_slice, comp)
    return primal_out, tangent_out

def _int1e_jvp_r0(mol, mol_t, intor):
    s1 = -intor2c(mol, intor, comp=3)
    aoslices = mol.aoslice_by_atom()[:,2:4]
    grad = _fill_grad_r0(s1, aoslices)
    tangent_out = _int1e_dot_grad_tangent_r0(grad, mol_t.coords)
    return tangent_out

@jit
def _fill_grad_r0(eri1, aoslices):
    nao = eri1.shape[-1]
    shape = [1,] * eri1.ndim
    shape[1] = nao
    idx = np.arange(nao)
    idx = idx.reshape(shape)
    def body(slices):
        p0, p1 = slices[:]
        mask = (idx >= p0) & (idx < p1)
        return np.where(mask, eri1, np.array(0, dtype=eri1.dtype))
    grad = vmap(body)(aoslices)
    return grad

@jit
def _int1e_dot_grad_tangent_r0(grad, tangent):
    tangent_out = np.einsum('nxij,nx->ij', grad, tangent)
    tangent_out += tangent_out.T
    return tangent_out

def _gen_int1e_jvp_r0(mol, mol_t, intor_a, intor_b, rc_deriv=None, hermi=0):
    coords_t = mol_t.coords
    #atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()[:,2:4]
    nao = mol.nao

    s1a = -getints2c_rc(mol, intor_a, hermi=0, rc_deriv=rc_deriv).reshape(3,-1,nao,nao)
    s1b = None
    if hermi == 0:
        order_a = int1e_get_dr_order(intor_b)[0]
        s1b = -getints2c_rc(mol, intor_b, hermi=0, rc_deriv=rc_deriv)
        s1b = s1b.reshape(3**order_a,3,-1,nao,nao).transpose(1,0,2,3,4).reshape(3,-1,nao,nao)
    #jvp = np.zeros(s1a.shape[1:])
    #for k, ia in enumerate(atmlst):
    #    p0, p1 = aoslices[ia,2:]
    #    ta = np.einsum('xyij,x->yij', s1a[...,p0:p1,:], coords_t[k])
    #    tb = np.einsum('xyij,x->yij', s1b[...,p0:p1], coords_t[k])
    #    jvp = ops.index_add(jvp, ops.index[:,p0:p1], ta)
    #    jvp = ops.index_add(jvp, ops.index[:,:,p0:p1], tb)
    idx = np.arange(nao)
    grad = _gen_int1e_fill_grad_r0(s1a, aoslices, idx[None,None,:,None])
    if hermi == 0:
        grad = grad + _gen_int1e_fill_grad_r0(s1b, aoslices, idx[None,None,None,:])
    if rc_deriv is not None:
        grad_rc = -s1a
        if hermi == 0:
            grad_rc -= s1b
        grad = ops.index_add(grad, ops.index[rc_deriv], grad_rc)
    jvp = np.einsum('nxyij,nx->yij', grad, coords_t)
    if hermi == 1:
        jvp += jvp.transpose(0,2,1)
    return jvp

@jit
def _gen_int1e_fill_grad_r0(s1, aoslices, idx):
    def body(slices):
        p0, p1 = slices[:]
        mask = (idx >= p0) & (idx < p1)
        grad = np.where(mask, s1, np.array(0, dtype=s1.dtype))
        return grad
    grad = vmap(body)(aoslices)
    return grad

def _int1e_r_jvp_r0(mol, mol_t, intor):
    coords_t = mol_t.coords
    atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()
    nao = mol.nao
    s1 = -intor2c(mol, intor).reshape(-1,3,nao,nao)
    grad = [numpy.zeros_like(s1) for ia in atmlst]
    grad = numpy.asarray(grad)
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices [ia,2:]
        grad[k,...,p0:p1] = s1[...,p0:p1]
    tangent_out = _int1e_r_dot_grad_tangent_r0(grad, coords_t)
    return tangent_out

@jit
def _int1e_r_dot_grad_tangent_r0(grad, tangent):
    tangent_out = np.einsum('npxij,nx->pij', grad, tangent)
    tangent_out += tangent_out.transpose(0,2,1)
    return tangent_out

def _int1e_nuc_jvp_rc(mol, mol_t, intor):
    coords_t = mol_t.coords
    atmlst = range(mol.natm)
    nao = mol.nao

    ecp_intor = 'ECP' in intor
    if ecp_intor:
        if not mol.has_ecp():
            return 0
        else:
            ecp_atoms = set(mol._ecpbas[:,ATOM_OF])

    grad = np.zeros((mol.natm,3,nao,nao))
    for k, ia in enumerate(atmlst):
        with mol.with_rinv_at_nucleus(ia):
            if not ecp_intor:
                vrinv = getints2c_rc(mol, intor, comp=3, rc_deriv=ia)
                vrinv *= -mol.atom_charge(ia)
            else:
                if ia in ecp_atoms:
                    vrinv = getints2c_rc(mol, intor, comp=3, rc_deriv=ia)
                else:
                    vrinv = 0
        grad = ops.index_update(grad, ops.index[k], vrinv)
    tangent_out = _int1e_dot_grad_tangent_r0(grad, coords_t)
    return tangent_out

def _gen_int1e_nuc_jvp_rc(mol, mol_t, intor_a, intor_b, hermi=0):
    coords_t = mol_t.coords
    atmlst = range(mol.natm)
    nao = mol.nao
    _, comp = _get_intor_and_comp(intor_a)
    grad = np.zeros((mol.natm,3,comp//3,nao,nao), dtype=float)
    for k, ia in enumerate(atmlst):
        with mol.with_rinv_at_nucleus(ia):
            vrinv = getints2c_rc(mol, intor_a, rc_deriv=ia).reshape(3,-1,nao,nao)
            if hermi == 0:
                order_a = int1e_get_dr_order(intor_b)[0]
                s1b = getints2c_rc(mol, intor_b, rc_deriv=ia)
                s1b = s1b.reshape(3**order_a,3,-1,nao,nao).transpose(1,0,2,3,4)
                vrinv += s1b.reshape(3,-1,nao,nao)
            if 'ECP' not in intor_a:
                vrinv *= -mol.atom_charge(ia)
        grad = ops.index_update(grad, ops.index[k], vrinv)
    jvp = np.einsum('nxyij,nx->yij', grad, coords_t)
    if hermi == 1:
        jvp = jvp + jvp.transpose(0,2,1)
    return jvp

def _int1e_jvp_cs(mol, mol_t, intor, shls_slice, comp, hermi):
    ctr_coeff = mol.ctr_coeff
    ctr_coeff_t = mol_t.ctr_coeff

    mol1 = get_fakemol_cs(mol)
    mol1._atm[:,pyscf_mole.CHARGE_OF] = 0 # set nuclear charge to zero

    nao = mol.nao
    nao1 = mol1.nao
    nbas = mol.nbas
    nbas1 = mol1.nbas

    intor = mol._add_suffix(intor)
    intor, comp = _get_intor_and_comp(intor)
    if 'ECP' in intor:
        assert mol._ecp is not None
        bas = numpy.vstack((mol._bas, mol._ecpbas))
    else:
        bas = mol._bas
    atmc, basc, envc = pyscf_mole.conc_env(mol1._atm, mol1._bas, mol1._env,
                                           mol._atm, bas, mol._env)
    if 'ECP' in intor:
        envc[pyscf_mole.AS_ECPBAS_OFFSET] = nbas1 + nbas
        envc[pyscf_mole.AS_NECPBAS] = len(mol._ecpbas)

    _, cs_of, _ = setup_ctr_coeff(mol)

    def _fill_grad_bra():
        shls_slice = (0, nbas1, nbas1, nbas1+nbas)
        s = moleintor.getints(intor, atmc, basc, envc, shls_slice)
        s = s.reshape(comp, nao1, nao)
        grad = numpy.zeros((comp,len(ctr_coeff),nao,nao), dtype=s.dtype)

        off = 0
        ibas = 0
        for i in range(nbas):
            l = mol._bas[i,pyscf_mole.ANG_OF]
            if mol.cart:
                nl = (l+1)*(l+2)//2
            else:
                nl = 2*l + 1
            nprim = mol._bas[i,pyscf_mole.NPRIM_OF]
            nctr = mol._bas[i,pyscf_mole.NCTR_OF]
            g = s[:,off:(off+nprim*nl)].reshape(comp,nprim,-1,nao)
            for j in range(nctr):
                grad[:,(cs_of[i]+j*nprim):(cs_of[i]+(j+1)*nprim),ibas:(ibas+nl)] += g
                ibas += nl
            off += nprim*nl
        s = None
        return grad

    def _fill_grad_ket():
        shls_slice = (nbas1, nbas1+nbas, 0, nbas1)
        s = moleintor.getints(intor, atmc, basc, envc, shls_slice)
        s = s.reshape(comp, nao, nao1)
        grad = numpy.zeros((comp,len(ctr_coeff),nao,nao), dtype=s.dtype)

        off = 0
        ibas = 0
        for i in range(nbas):
            l = mol._bas[i,pyscf_mole.ANG_OF]
            if mol.cart:
                nl = (l+1)*(l+2)//2
            else:
                nl = 2*l + 1
            nprim = mol._bas[i,pyscf_mole.NPRIM_OF]
            nctr = mol._bas[i,pyscf_mole.NCTR_OF]
            g = s[:,:,off:(off+nprim*nl)].reshape(comp,nao,nprim,-1)
            g = g.transpose(0,2,1,3)
            for j in range(nctr):
                grad[:,(cs_of[i]+j*nprim):(cs_of[i]+(j+1)*nprim),:,ibas:(ibas+nl)] += g
                ibas += nl
            off += nprim*nl
        s = None
        return grad

    grad = _fill_grad_bra()
    if hermi == 1:
        grad += grad.transpose(0,1,3,2)
    elif hermi == 0:
        grad += _fill_grad_ket()
    else:
        raise NotImplementedError

    tangent_out = np.einsum('cxij,x->cij', grad, ctr_coeff_t)
    if comp == 1:
        tangent_out = tangent_out[0]
    return tangent_out

# FIXME allow higher order derivatives
def _int1e_jvp_exp(mol, mol_t, intor, shls_slice, comp, hermi):
    mol1 = get_fakemol_exp(mol)
    mol1._atm[:,pyscf_mole.CHARGE_OF] = 0 # set nuclear charge to zero
    if intor.endswith('_sph'):
        intor = intor.replace('_sph', '_cart')
        cart = False
    else:
        cart = True
        intor = mol._add_suffix(intor, cart=True)
    intor, comp = _get_intor_and_comp(intor)

    nbas = len(mol._bas)
    nbas1 = len(mol1._bas)
    if 'ECP' in intor:
        assert mol._ecp is not None
        bas = numpy.vstack((mol._bas, mol._ecpbas))
    else:
        bas = mol._bas
    atmc, basc, envc = pyscf_mole.conc_env(mol1._atm, mol1._bas, mol1._env,
                                           mol._atm, bas, mol._env)
    if 'ECP' in intor:
        envc[pyscf_mole.AS_ECPBAS_OFFSET] = nbas1 + nbas
        envc[pyscf_mole.AS_NECPBAS] = len(mol._ecpbas)

    nao = pyscf_mole.nao_cart(mol)
    es, es_of, _env_of = setup_exp(mol)

    def _fill_grad_bra():
        shls_slice = (0, nbas1, nbas1, nbas1+nbas)
        s = moleintor.getints(intor, atmc, basc, envc, shls_slice)
        s = s.reshape(comp, -1, nao)
        grad = numpy.zeros((comp,len(es),nao,nao), dtype=s.dtype)

        off = 0
        ibas = 0
        for i in range(nbas):
            ioff = es_of[i]

            l = mol._bas[i,pyscf_mole.ANG_OF]
            nl = (l+1)*(l+2)//2
            nl1 = (l+3)*(l+4)//2
            nprim = mol._bas[i,pyscf_mole.NPRIM_OF]
            nctr = mol._bas[i,pyscf_mole.NCTR_OF]
            ptr_ctr_coeff = mol._bas[i,pyscf_mole.PTR_COEFF]
            g = s[:,off:off+nprim*nl1].reshape(comp, nprim, nl1, nao)

            xyz = get_bas_label(l)
            xyz1 = get_bas_label(l+2)
            for k in range(nctr):
                for j in range(nprim):
                    c = mol._env[ptr_ctr_coeff + k*nprim + j]
                    if l == 0:
                        c *= _S_NORM
                    elif l == 1:
                        c *= _P_NORM
                    jbas = ibas
                    for orb in xyz:
                        idx_x = xyz1.index(promote_xyz(orb, 'x', 2))
                        idx_y = xyz1.index(promote_xyz(orb, 'y', 2))
                        idx_z = xyz1.index(promote_xyz(orb, 'z', 2))
                        gc = -(g[:,j,idx_x] + g[:,j,idx_y] + g[:,j,idx_z]) *  c
                        grad[:,ioff+j,jbas] += gc
                        jbas += 1
                ibas += nl
            off += nprim * nl1
        s = None
        return grad

    def _fill_grad_ket():
        shls_slice = (nbas1, nbas1+nbas, 0, nbas1)
        s = moleintor.getints(intor, atmc, basc, envc, shls_slice)
        s = s.reshape(comp, nao, -1)
        grad = numpy.zeros((comp,len(es),nao,nao), dtype=s.dtype)

        off = 0
        ibas = 0
        for i in range(nbas):
            ioff = es_of[i]

            l = mol._bas[i,pyscf_mole.ANG_OF]
            nl = (l+1)*(l+2)//2
            nl1 = (l+3)*(l+4)//2
            nprim = mol._bas[i,pyscf_mole.NPRIM_OF]
            nctr = mol._bas[i,pyscf_mole.NCTR_OF]
            ptr_ctr_coeff = mol._bas[i,pyscf_mole.PTR_COEFF]
            g = s[:,:,off:off+nprim*nl1].reshape(comp, nao, nprim, nl1)
            g = g.transpose(0,2,1,3)

            xyz = get_bas_label(l)
            xyz1 = get_bas_label(l+2)
            for k in range(nctr):
                for j in range(nprim):
                    c = mol._env[ptr_ctr_coeff + k*nprim + j]
                    if l == 0:
                        c *= _S_NORM
                    elif l == 1:
                        c *= _P_NORM
                    jbas = ibas
                    for orb in xyz:
                        idx_x = xyz1.index(promote_xyz(orb, 'x', 2))
                        idx_y = xyz1.index(promote_xyz(orb, 'y', 2))
                        idx_z = xyz1.index(promote_xyz(orb, 'z', 2))
                        gc = -(g[:,j,:,idx_x] + g[:,j,:,idx_y] + g[:,j,:,idx_z]) *  c
                        grad[:,ioff+j,:,jbas] += gc
                        jbas += 1
                ibas += nl
            off += nprim * nl1
        s = None
        return grad

    grad = _fill_grad_bra()
    if hermi == 1:
        grad += grad.transpose(0,1,3,2)
    elif hermi == 0:
        grad += _fill_grad_ket()
    else:
        raise NotImplementedError

    tangent_out = np.einsum('cxij,x->cij', grad, mol_t.exp)
    if not mol.cart or not cart:
        c2s = np.asarray(mol.cart2sph_coeff())
        tangent_out = np.einsum('pi,cpq,qj->cij', c2s, tangent_out, c2s)
    if comp == 1:
        tangent_out = tangent_out[0]
    return tangent_out

def _int2e_jvp_r0(mol, mol_t, intor):
    eri1 = -intor4c(mol, intor, comp=None, aosym='s1')
    aoslices = mol.aoslice_by_atom()[:,2:4]
    grad = _fill_grad_r0(eri1, aoslices)
    return _int2e_dot_grad_tangent_r0(grad, mol_t.coords)

@jit
def _int2e_dot_grad_tangent_r0(grad, tangent):
    tangent_out = np.einsum('nxijkl,nx->ijkl', grad, tangent)
    tangent_out += tangent_out.transpose(1,0,2,3)
    tangent_out += tangent_out.transpose(2,3,0,1)
    return tangent_out

def _gen_int2e_jvp_r0(mol, mol_t, intors):
    coords_t = mol_t.coords
    nao = mol.nao
    intor_a, intor_b, intor_c, intor_d = intors
    eri1_a = -intor4c(mol, intor_a, aosym='s1').reshape(3,-1,nao,nao,nao,nao)
    if intor_b:
        orders = int2e_get_dr_order(intor_b)
        off = 3**orders[0]
        eri1_b = -intor4c(mol, intor_b, aosym='s1')
        eri1_b = eri1_b.reshape(off,3,-1,nao,nao,nao,nao)
        eri1_b = eri1_b.transpose(1,0,2,3,4,5,6).reshape(3,-1,nao,nao,nao,nao)
    else:
        eri1_b = eri1_a.transpose(0,1,3,2,4,5)

    orders = int2e_get_dr_order(intor_c)
    off = 3**(orders[0]+orders[1])
    eri1_c = -intor4c(mol, intor_c, aosym='s1')
    eri1_c = eri1_c.reshape(off,3,-1,nao,nao,nao,nao)
    eri1_c = eri1_c.transpose(1,0,2,3,4,5,6).reshape(3,-1,nao,nao,nao,nao)
    if intor_d:
        orders = int2e_get_dr_order(intor_d)
        off = 3**(orders[0]+orders[1]+orders[2])
        eri1_d = -intor4c(mol, intor_d, aosym='s1')
        eri1_d = eri1_d.reshape(off,3,-1,nao,nao,nao,nao)
        eri1_d = eri1_d.transpose(1,0,2,3,4,5,6).reshape(3,-1,nao,nao,nao,nao)
    else:
        eri1_d = eri1_c.transpose(0,1,2,3,5,4)

    aoslices = mol.aoslice_by_atom()[:,2:4]
    idx = np.arange(nao)
    idx_a = idx[None,None,:,None,None,None]
    idx_b = idx[None,None,None,:,None,None]
    idx_c = idx[None,None,None,None,:,None]
    idx_d = idx[None,None,None,None,None,:]
    grad = _gen_int2e_fill_grad_r0(eri1_a, eri1_b, eri1_c, eri1_d, aoslices,
                                   idx_a, idx_b, idx_c, idx_d)
    jvp = np.einsum('nxyijkl,nx->yijkl', grad, coords_t)
    return jvp

@jit
def _gen_int2e_fill_grad_r0(eri1_a, eri1_b, eri1_c, eri1_d, aoslices,
                            idx_a, idx_b, idx_c, idx_d):
    def body(slices):
        p0, p1 = slices[:]
        mask_a = (idx_a >= p0) & (idx_a < p1)
        mask_b = (idx_b >= p0) & (idx_b < p1)
        mask_c = (idx_c >= p0) & (idx_c < p1)
        mask_d = (idx_d >= p0) & (idx_d < p1)
        grad_a = np.where(mask_a, eri1_a, np.array(0, dtype=eri1_a.dtype))
        grad_b = np.where(mask_b, eri1_b, np.array(0, dtype=eri1_b.dtype))
        grad_c = np.where(mask_c, eri1_c, np.array(0, dtype=eri1_c.dtype))
        grad_d = np.where(mask_d, eri1_d, np.array(0, dtype=eri1_d.dtype))
        grad = grad_a + grad_b + grad_c + grad_d
        return grad
    grad = vmap(body)(aoslices)
    return grad

@jit
def _int2e_dot_grad_tangent_s4(grad, tangent):
    tangent_out = np.einsum('cxijkl,x->cijkl', grad, tangent)
    tangent_out += tangent_out.transpose(0,2,1,3,4)
    tangent_out += tangent_out.transpose(0,3,4,1,2)
    return tangent_out

def _int2e_jvp_cs(mol, mol_t, intor, shls_slice, comp):
    ctr_coeff = mol.ctr_coeff
    ctr_coeff_t = mol_t.ctr_coeff

    mol1 = get_fakemol_cs(mol)

    nao = mol.nao
    nao1 = mol1.nao
    nbas = mol.nbas
    nbas1 = mol1.nbas

    atmc, basc, envc = pyscf_mole.conc_env(mol1._atm, mol1._bas, mol1._env,
                                           mol._atm, mol._bas, mol._env)

    intor = mol._add_suffix(intor)
    intor, comp = _get_intor_and_comp(intor)

    _, cs_of, _ = setup_ctr_coeff(mol)

    def _fill_grad0():
        shls_slice = (0, nbas1,
                      nbas1, nbas1+nbas,
                      nbas1, nbas1+nbas,
                      nbas1, nbas1+nbas)
        eri = moleintor.getints(intor, atmc, basc, envc, shls_slice, comp)
        eri = eri.reshape(comp, nao1, nao, nao, nao)
        grad = numpy.zeros((comp,len(ctr_coeff),nao,nao,nao,nao), dtype=eri.dtype)

        off = 0
        ibas = 0
        for i in range(nbas):
            l = mol._bas[i,pyscf_mole.ANG_OF]
            if mol.cart:
                nl = (l+1)*(l+2)//2
            else:
                nl = 2*l + 1
            nprim = mol._bas[i,pyscf_mole.NPRIM_OF]
            nctr = mol._bas[i,pyscf_mole.NCTR_OF]
            g = eri[:,off:(off+nprim*nl)].reshape(comp,nprim,-1,nao,nao,nao)
            for j in range(nctr):
                grad[:, (cs_of[i]+j*nprim):(cs_of[i]+(j+1)*nprim), ibas:(ibas+nl)] += g
                ibas += nl
            off += nprim*nl
        eri = None
        return grad

    def _fill_grad1():
        shls_slice = (nbas1, nbas1+nbas,
                      0, nbas1,
                      nbas1, nbas1+nbas,
                      nbas1, nbas1+nbas)
        eri = moleintor.getints(intor, atmc, basc, envc, shls_slice, comp)
        eri = eri.reshape(comp, nao, nao1, nao, nao)
        grad = numpy.zeros((comp,len(ctr_coeff),nao,nao,nao,nao), dtype=eri.dtype)

        off = 0
        ibas = 0
        for i in range(nbas):
            l = mol._bas[i,pyscf_mole.ANG_OF]
            if mol.cart:
                nl = (l+1)*(l+2)//2
            else:
                nl = 2*l + 1
            nprim = mol._bas[i,pyscf_mole.NPRIM_OF]
            nctr = mol._bas[i,pyscf_mole.NCTR_OF]
            g = eri[:,:,off:(off+nprim*nl)].reshape(comp,nao,nprim,-1,nao,nao)
            g = g.transpose(0,2,1,3,4,5)
            for j in range(nctr):
                grad[:,(cs_of[i]+j*nprim):(cs_of[i]+(j+1)*nprim),:,ibas:(ibas+nl)] += g
                ibas += nl
            off += nprim*nl
        eri = None
        return grad

    def _fill_grad2():
        shls_slice = (nbas1, nbas1+nbas,
                      nbas1, nbas1+nbas,
                      0, nbas1,
                      nbas1, nbas1+nbas)
        eri = moleintor.getints(intor, atmc, basc, envc, shls_slice, comp)
        eri = eri.reshape(comp, nao, nao, nao1, nao)
        grad = numpy.zeros((comp,len(ctr_coeff),nao,nao,nao,nao), dtype=eri.dtype)

        off = 0
        ibas = 0
        for i in range(nbas):
            l = mol._bas[i,pyscf_mole.ANG_OF]
            if mol.cart:
                nl = (l+1)*(l+2)//2
            else:
                nl = 2*l + 1
            nprim = mol._bas[i,pyscf_mole.NPRIM_OF]
            nctr = mol._bas[i,pyscf_mole.NCTR_OF]
            g = eri[:,:,:,off:(off+nprim*nl)].reshape(comp,nao,nao,nprim,-1,nao)
            g = g.transpose(0,3,1,2,4,5)
            for j in range(nctr):
                grad[:,(cs_of[i]+j*nprim):(cs_of[i]+(j+1)*nprim),:,:,ibas:(ibas+nl)] += g
                ibas += nl
            off += nprim*nl
        eri = None
        return grad

    def _fill_grad3():
        shls_slice = (nbas1, nbas1+nbas,
                      nbas1, nbas1+nbas,
                      nbas1, nbas1+nbas,
                      0, nbas1)
        eri = moleintor.getints(intor, atmc, basc, envc, shls_slice, comp)
        eri = eri.reshape(comp, nao, nao, nao, nao1)
        grad = numpy.zeros((comp,len(ctr_coeff),nao,nao,nao,nao), dtype=eri.dtype)

        off = 0
        ibas = 0
        for i in range(nbas):
            l = mol._bas[i,pyscf_mole.ANG_OF]
            if mol.cart:
                nl = (l+1)*(l+2)//2
            else:
                nl = 2*l + 1
            nprim = mol._bas[i,pyscf_mole.NPRIM_OF]
            nctr = mol._bas[i,pyscf_mole.NCTR_OF]
            g = eri[:,:,:,:,off:(off+nprim*nl)].reshape(comp,nao,nao,nao,nprim,-1)
            g = g.transpose(0,4,1,2,3,5)
            for j in range(nctr):
                grad[:,(cs_of[i]+j*nprim):(cs_of[i]+(j+1)*nprim),:,:,:,ibas:(ibas+nl)] += g
                ibas += nl
            off += nprim*nl
        eri = None
        return grad

    grad = _fill_grad0()
    if comp == 1:
        tangent_out = _int2e_dot_grad_tangent_s4(grad, ctr_coeff_t)
    else:
        grad += _fill_grad1()
        grad += _fill_grad2()
        grad += _fill_grad3()
        tangent_out  = np.einsum('cxijkl,x->cijkl', grad, ctr_coeff_t)

    if comp == 1:
        tangent_out = tangent_out[0]
    return tangent_out

# FIXME allow higher order derivatives
def _int2e_jvp_exp(mol, mol_t, intor, shls_slice, comp):
    mol1 = get_fakemol_exp(mol)
    mol1._atm[:,pyscf_mole.CHARGE_OF] = 0 # set nuclear charge to zero
    if intor.endswith('_sph'):
        intor = intor.replace('_sph', '_cart')
        cart = False
    else:
        cart = True
        intor = mol._add_suffix(intor, cart=True)
    intor, comp = _get_intor_and_comp(intor)

    nbas = len(mol._bas)
    nbas1 = len(mol1._bas)
    atmc, basc, envc = pyscf_mole.conc_env(mol1._atm, mol1._bas, mol1._env,
                                     mol._atm, mol._bas, mol._env)

    nao = pyscf_mole.nao_cart(mol)
    es, es_of, _env_of = setup_exp(mol)

    def _fill_grad0():
        shls_slice = (0, nbas1,
                      nbas1, nbas1+nbas,
                      nbas1, nbas1+nbas,
                      nbas1, nbas1+nbas)
        eri = moleintor.getints(intor, atmc, basc, envc, shls_slice)
        eri = eri.reshape(comp, -1, nao, nao, nao)
        grad = numpy.zeros((comp, len(es), nao, nao, nao, nao), dtype=eri.dtype)

        off = 0
        ibas = 0
        for i in range(nbas):
            ioff = es_of[i]

            l = mol._bas[i,pyscf_mole.ANG_OF]
            nl = (l+1)*(l+2)//2
            nl1 = (l+3)*(l+4)//2
            nprim = mol._bas[i,pyscf_mole.NPRIM_OF]
            nctr = mol._bas[i,pyscf_mole.NCTR_OF]
            ptr_ctr_coeff = mol._bas[i,pyscf_mole.PTR_COEFF]
            g = eri[:,off:off+nprim*nl1].reshape(comp, nprim, nl1, nao, nao, nao)

            xyz = get_bas_label(l)
            xyz1 = get_bas_label(l+2)
            for k in range(nctr):
                for j in range(nprim):
                    c = mol._env[ptr_ctr_coeff + k*nprim + j]
                    if l == 0:
                        c *= _S_NORM
                    elif l == 1:
                        c *= _P_NORM
                    jbas = ibas
                    for orb in xyz:
                        idx_x = xyz1.index(promote_xyz(orb, 'x', 2))
                        idx_y = xyz1.index(promote_xyz(orb, 'y', 2))
                        idx_z = xyz1.index(promote_xyz(orb, 'z', 2))
                        gc = -(g[:,j,idx_x] + g[:,j,idx_y] + g[:,j,idx_z]) *  c
                        grad[:, ioff+j, jbas] += gc
                        jbas += 1
                ibas += nl
            off += nprim * nl1
        eri = None
        return grad

    def _fill_grad1():
        shls_slice = (nbas1, nbas1+nbas,
                      0, nbas1,
                      nbas1, nbas1+nbas,
                      nbas1, nbas1+nbas)
        eri = moleintor.getints(intor, atmc, basc, envc, shls_slice)
        eri = eri.reshape(comp, nao, -1, nao, nao)
        grad = numpy.zeros((comp, len(es), nao, nao, nao, nao), dtype=eri.dtype)

        off = 0
        ibas = 0
        for i in range(nbas):
            ioff = es_of[i]

            l = mol._bas[i,pyscf_mole.ANG_OF]
            nl = (l+1)*(l+2)//2
            nl1 = (l+3)*(l+4)//2
            nprim = mol._bas[i,pyscf_mole.NPRIM_OF]
            nctr = mol._bas[i,pyscf_mole.NCTR_OF]
            ptr_ctr_coeff = mol._bas[i,pyscf_mole.PTR_COEFF]
            g = eri[:,:,off:off+nprim*nl1].reshape(comp, nao, nprim, nl1, nao, nao)
            g = numpy.moveaxis(g, 2, 1)

            xyz = get_bas_label(l)
            xyz1 = get_bas_label(l+2)
            for k in range(nctr):
                for j in range(nprim):
                    c = mol._env[ptr_ctr_coeff + k*nprim + j]
                    if l == 0:
                        c *= _S_NORM
                    elif l == 1:
                        c *= _P_NORM
                    jbas = ibas
                    for orb in xyz:
                        idx_x = xyz1.index(promote_xyz(orb, 'x', 2))
                        idx_y = xyz1.index(promote_xyz(orb, 'y', 2))
                        idx_z = xyz1.index(promote_xyz(orb, 'z', 2))
                        gc = -(g[:,j,:,idx_x] + g[:,j,:,idx_y] + g[:,j,:,idx_z]) *  c
                        grad[:, ioff+j, :, jbas] += gc
                        jbas += 1
                ibas += nl
            off += nprim * nl1
        eri = None
        return grad

    def _fill_grad2():
        shls_slice = (nbas1, nbas1+nbas,
                      nbas1, nbas1+nbas,
                      0, nbas1,
                      nbas1, nbas1+nbas)
        eri = moleintor.getints(intor, atmc, basc, envc, shls_slice)
        eri = eri.reshape(comp, nao, nao, -1, nao)
        grad = numpy.zeros((comp, len(es), nao, nao, nao, nao), dtype=eri.dtype)

        off = 0
        ibas = 0
        for i in range(nbas):
            ioff = es_of[i]

            l = mol._bas[i,pyscf_mole.ANG_OF]
            nl = (l+1)*(l+2)//2
            nl1 = (l+3)*(l+4)//2
            nprim = mol._bas[i,pyscf_mole.NPRIM_OF]
            nctr = mol._bas[i,pyscf_mole.NCTR_OF]
            ptr_ctr_coeff = mol._bas[i,pyscf_mole.PTR_COEFF]
            g = eri[:,:,:,off:off+nprim*nl1].reshape(comp, nao, nao, nprim, nl1, nao)
            g = numpy.moveaxis(g, 3, 1)

            xyz = get_bas_label(l)
            xyz1 = get_bas_label(l+2)
            for k in range(nctr):
                for j in range(nprim):
                    c = mol._env[ptr_ctr_coeff + k*nprim + j]
                    if l == 0:
                        c *= _S_NORM
                    elif l == 1:
                        c *= _P_NORM
                    jbas = ibas
                    for orb in xyz:
                        idx_x = xyz1.index(promote_xyz(orb, 'x', 2))
                        idx_y = xyz1.index(promote_xyz(orb, 'y', 2))
                        idx_z = xyz1.index(promote_xyz(orb, 'z', 2))
                        gc = -(g[:,j,:,:,idx_x] + g[:,j,:,:,idx_y] + g[:,j,:,:,idx_z]) *  c
                        grad[:, ioff+j, :, :, jbas] += gc
                        jbas += 1
                ibas += nl
            off += nprim * nl1
        eri = None
        return grad

    def _fill_grad3():
        shls_slice = (nbas1, nbas1+nbas,
                      nbas1, nbas1+nbas,
                      nbas1, nbas1+nbas,
                      0, nbas1)
        eri = moleintor.getints(intor, atmc, basc, envc, shls_slice)
        eri = eri.reshape(comp, nao, nao, nao, -1)
        grad = numpy.zeros((comp, len(es), nao, nao, nao, nao), dtype=eri.dtype)

        off = 0
        ibas = 0
        for i in range(nbas):
            ioff = es_of[i]

            l = mol._bas[i,pyscf_mole.ANG_OF]
            nl = (l+1)*(l+2)//2
            nl1 = (l+3)*(l+4)//2
            nprim = mol._bas[i,pyscf_mole.NPRIM_OF]
            nctr = mol._bas[i,pyscf_mole.NCTR_OF]
            ptr_ctr_coeff = mol._bas[i,pyscf_mole.PTR_COEFF]
            g = eri[:,:,:,:,off:off+nprim*nl1].reshape(comp, nao, nao, nao, nprim, nl1)
            g = numpy.moveaxis(g, 4, 1)

            xyz = get_bas_label(l)
            xyz1 = get_bas_label(l+2)
            for k in range(nctr):
                for j in range(nprim):
                    c = mol._env[ptr_ctr_coeff + k*nprim + j]
                    if l == 0:
                        c *= _S_NORM
                    elif l == 1:
                        c *= _P_NORM
                    jbas = ibas
                    for orb in xyz:
                        idx_x = xyz1.index(promote_xyz(orb, 'x', 2))
                        idx_y = xyz1.index(promote_xyz(orb, 'y', 2))
                        idx_z = xyz1.index(promote_xyz(orb, 'z', 2))
                        gc = -(g[:,j,:,:,:,idx_x] + g[:,j,:,:,:,idx_y] + g[:,j,:,:,:,idx_z]) *  c
                        grad[:, ioff+j, :, :, :, jbas] += gc
                        jbas += 1
                ibas += nl
            off += nprim * nl1
        eri = None
        return grad

    grad = _fill_grad0()
    if comp == 1:
        tangent_out = _int2e_dot_grad_tangent_s4(grad, mol_t.exp)
    else:
        grad += _fill_grad1()
        grad += _fill_grad2()
        grad += _fill_grad3()
        tangent_out = np.einsum('cxijkl,x->cijkl', grad, mol_t.exp)

    if not mol.cart or not cart:
        c2s = np.asarray(mol.cart2sph_coeff())
        tangent_out = _int2e_c2s(tangent_out, c2s)
    if comp == 1:
        tangent_out = tangent_out[0]
    return tangent_out

def _int2e_c2s(eris_cart, c2s):
    eris_sph = np.einsum('pi,qj,cpqrs,rk,sl->cijkl',
                         c2s, c2s, eris_cart, c2s, c2s)
    return eris_sph
