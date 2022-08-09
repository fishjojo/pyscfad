import warnings
from functools import partial
import ctypes
import numpy
from jax import jit
from jax import vmap
from pyscf import ao2mo
from pyscf.gto import mole, moleintor
from pyscf.gto.mole import Mole
from pyscf.gto.moleintor import _get_intor_and_comp
from pyscfad.lib import numpy as np
from pyscfad.lib import ops, custom_jvp
from ._mole_helper import uncontract, setup_exp, setup_ctr_coeff

SET_RC = ["rinv",]

@partial(custom_jvp, nondiff_argnums=(0,3,4))
def intor_cross(intor, mol1, mol2, comp=None, grids=None):
    return mole.intor_cross(intor, mol1, mol2, comp=comp, grids=grids)

@intor_cross.defjvp
def intor_cross_jvp(intor, comp, grids,
                    primals, tangents):
    mol1, mol2 = primals
    mol1_t, mol2_t = tangents

    primal_out = intor_cross(intor, mol1, mol2, comp=comp, grids=grids)
    tangent_out = np.zeros_like(primal_out)

    if mol1.coords is not None:
        nao1 = mol1.nao
        aoslices1 = mol1.aoslice_by_atom()[:,2:4]
        intor_ip_bra, _ = _int1e_dr1_name(intor)
        s1a = -intor_cross(intor_ip_bra, mol1, mol2, comp=None, grids=grids).reshape(3,-1,nao1,nao1)

        idx1 = np.arange(nao1)
        grad1 = _gen_int1e_fill_grad_r0(s1a, aoslices1, idx1[None,None,:,None])
        tangent_out += np.einsum('nxyij,nx->yij', grad1, mol1_t.coords).reshape(primal_out.shape)

    if mol2.coords is not None:
        nao2 = mol2.nao
        aoslices2 = mol2.aoslice_by_atom()[:,2:4]
        _, intor_ip_ket = _int1e_dr1_name(intor)
        s1b = -intor_cross(intor_ip_ket, mol1, mol2, comp=None, grids=grids)

        order_a = _int1e_get_dr_order(intor_ip_ket)[0]
        s1b = s1b.reshape(3**order_a,3,-1,nao2,nao2).transpose(1,0,2,3,4).reshape(3,-1,nao2,nao2)
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

def getints(mol, intor, shls_slice=None,
            comp=None, hermi=0, aosym='s1', out=None):
    if intor.endswith("_spinor"):
        raise NotImplementedError('Spinors are not supported for AD.')
    if hermi == 2:
        hermi = 0
        msg = f'Anti-hermitian symmetry is not supported. Setting hermi = {hermi}.'
        warnings.warn(msg)
    if aosym != 's1':
        aosym = 's1'
        msg = f'AO symmetry is not supported. Setting aosym = {aosym}.'
        warnings.warn(msg)

    if (intor.startswith("int1e") or
        intor.startswith("int2c2e") or
        intor.startswith('ECP')):
        return getints2c(mol, intor, shls_slice, comp, hermi, aosym, out)
    elif intor.startswith("int2e"):
        return getints4c(mol, intor, shls_slice, comp, aosym, out)
    else:
        raise NotImplementedError

def _int1e_dr1_name(intor):
    if 'sph' in intor:
        suffix = '_sph'
    elif 'cart' in intor:
        suffix = '_cart'
    else:
        suffix = ''
    fname = intor.replace('_sph', '').replace('_cart', '')

    if fname[-4:-2] == 'dr':
        orders = [int(fname[-2]), int(fname[-1])]
        intor_ip_bra = fname[:-2] + str(orders[0]+1) + str(orders[1]) + suffix
        intor_ip_ket = fname[:-2] + str(orders[0]) + str(orders[1]+1) + suffix
    else:
        intor_ip_bra = fname + '_dr10' + suffix
        intor_ip_ket = fname + '_dr01' + suffix
    return intor_ip_bra, intor_ip_ket

def _int1e_get_dr_order(intor):
    fname = intor.replace('_sph', '').replace('_cart', '')
    if fname[-4:-2] == 'dr':
        orders = [int(fname[-2]), int(fname[-1])]
    else:
        orders = [0, 0]
    return orders

def _int2e_dr1_name(intor):
    if 'sph' in intor:
        suffix = '_sph'
    elif 'cart' in intor:
        suffix = '_cart'
    else:
        suffix = ''
    fname = intor.replace('_sph', '').replace('_cart', '')

    if fname[-6:-4] == 'dr':
        orders = _int2e_get_dr_order(intor)
        str1 = str(orders[0]+1) + str(orders[1]) + str(orders[2]) + str(orders[3])
        str2 = str(orders[0]) + str(orders[1]+1) + str(orders[2]) + str(orders[3])
        str3 = str(orders[0]) + str(orders[1]) + str(orders[2]+1) + str(orders[3])
        str4 = str(orders[0]) + str(orders[1]) + str(orders[2]) + str(orders[3]+1)
        intor1 = fname[:-4] + str1 + suffix
        intor2 = fname[:-4] + str2 + suffix
        intor3 = fname[:-4] + str3 + suffix
        intor4 = fname[:-4] + str4 + suffix
    else:
        intor1 = fname + '_dr1000' + suffix
        intor2 = fname + '_dr0100' + suffix
        intor3 = fname + '_dr0010' + suffix
        intor4 = fname + '_dr0001' + suffix
    return intor1, intor2, intor3, intor4

def _int2e_get_dr_order(intor):
    fname = intor.replace('_sph', '').replace('_cart', '')
    if fname[-6:-4] == 'dr':
        orders = [int(fname[-4]), int(fname[-3]), int(fname[-2]), int(fname[-1])]
    else:
        orders = [0,] * 4
    return orders

def getints2c_rc(mol, intor, shls_slice=None, comp=None,
                 hermi=0, aosym='s1', out=None, rc_deriv=None):
    if rc_deriv is None or not any(rc in intor for rc in SET_RC):
        return getints2c(mol, intor, shls_slice, comp, hermi, aosym, out)
    else:
        return _getints2c_rc(mol, intor, shls_slice, comp, hermi, aosym, out, rc_deriv)

@partial(custom_jvp, nondiff_argnums=tuple(range(1,8)))
def _getints2c_rc(mol, intor, shls_slice=None, comp=None,
                  hermi=0, aosym='s1', out=None, rc_deriv=None):
    return Mole.intor(mol, intor, comp=comp, hermi=hermi, aosym=aosym,
                      shls_slice=shls_slice, out=out)

@_getints2c_rc.defjvp
def _getints2c_rc_jvp(intor, shls_slice, comp, hermi, aosym, out, rc_deriv,
                      primals, tangents):
    if shls_slice is not None:
        raise NotImplementedError

    mol, = primals
    mol_t, = tangents
    primal_out = _getints2c_rc(mol, intor, shls_slice, comp,
                               hermi, aosym, out)
    tangent_out = np.zeros_like(primal_out)

    if mol.coords is not None:
        intor_ip_bra, intor_ip_ket = _int1e_dr1_name(intor)
        tangent_out += _gen_int1e_jvp_r0(mol, mol_t, intor_ip_bra, intor_ip_ket, rc_deriv)
    return primal_out, tangent_out

@partial(custom_jvp, nondiff_argnums=tuple(range(1,7)))
def getints2c(mol, intor, shls_slice=None, comp=None, hermi=0, aosym='s1', out=None):
    return Mole.intor(mol, intor, comp=comp, hermi=hermi, aosym=aosym,
                      shls_slice=shls_slice, out=out)

@getints2c.defjvp
def getints2c_jvp(intor, shls_slice, comp, hermi, aosym, out,
                  primals, tangents):
    if shls_slice is not None:
        msg = 'AD for integrals with subblock of shells are not supported yet.'
        raise NotImplementedError(msg)

    mol, = primals
    mol_t, = tangents

    primal_out = getints2c(mol, intor, shls_slice=shls_slice,
                           comp=comp, hermi=hermi, aosym=aosym, out=out)

    tangent_out = np.zeros_like(primal_out)
    fname = intor.replace('_sph', '').replace('_cart', '')
    if mol.coords is not None:
        intor_ip_bra = intor_ip_ket = intor_ip = None
        if intor.startswith("ECPscalar"):
            intor_ip = intor.replace("ECPscalar", "ECPscalar_ipnuc")
        elif fname == 'int1e_r':
            intor_ip = intor.replace('int1e_r', 'int1e_irp')
        elif fname.startswith("int1e") or fname.startswith("int2c2e"):
            intor_ip_bra, intor_ip_ket = _int1e_dr1_name(intor)
        else:
            raise NotImplementedError(f'Integral {intor} is not supported for AD.')

        if intor_ip_bra or intor_ip_ket:
            tangent_out += _gen_int1e_jvp_r0(mol, mol_t,
                                intor_ip_bra, intor_ip_ket, hermi=hermi).reshape(tangent_out.shape)
            if "nuc" in intor_ip_bra and "nuc" in intor_ip_ket:
                intor_ip_bra = intor_ip_bra.replace("nuc", "rinv")
                intor_ip_ket = intor_ip_ket.replace("nuc", "rinv")
                tangent_out += _gen_int1e_nuc_jvp_rc(mol, mol_t,
                                intor_ip_bra, intor_ip_ket, hermi=hermi).reshape(tangent_out.shape)
        elif fname == 'int1e_r':
            tangent_out += _int1e_r_jvp_r0(mol, mol_t, intor_ip)
        else:
            tangent_out += _int1e_jvp_r0(mol, mol_t, intor_ip)

        intor_ip = None
        if intor.startswith("ECPscalar"):
            intor_ip = intor.replace("ECPscalar", "ECPscalar_iprinv")
        if intor_ip:
            tangent_out += _int1e_nuc_jvp_rc(mol, mol_t, intor_ip)

    if mol.ctr_coeff is not None:
        tangent_out += _int1e_jvp_cs(mol, mol_t, intor)

    if mol.exp is not None:
        tangent_out += _int1e_jvp_exp(mol, mol_t, intor)
    return primal_out, tangent_out

@partial(custom_jvp, nondiff_argnums=tuple(range(1,6)))
def getints4c(mol, intor,
              shls_slice=None, comp=None, aosym='s1', out=None):
    if (shls_slice is None and aosym=='s1'
            and intor in ['int2e', 'int2e_sph', 'int2e_cart']):
        eri8 = Mole.intor(mol, intor, comp=comp, aosym='s8',
                          shls_slice=shls_slice, out=out)
        eri = ao2mo.restore(aosym, eri8, mol.nao)
        del eri8
    else:
        eri = Mole.intor(mol, intor, comp=comp, aosym=aosym,
                         shls_slice=shls_slice, out=out)
    return eri

@getints4c.defjvp
def getints4c_jvp(intor, shls_slice, comp, aosym, out,
                  primals, tangents):
    if shls_slice is not None or out is not None:
        raise NotImplementedError
    if aosym != 's1':
        raise NotImplementedError

    mol, = primals
    primal_out = getints4c(mol, intor, shls_slice, comp, aosym, out)

    mol_t, = tangents
    tangent_out = np.zeros_like(primal_out)

    fname = intor.replace('_sph', '').replace('_cart', '')
    if mol.coords is not None:
        intor1, intor2, intor3, intor4 = _int2e_dr1_name(intor)
        if fname[-6:-4] == 'dr':
            orders = _int2e_get_dr_order(intor)
            if orders[0] == 0 and orders[1] == 0:
                intor2 = None
            if orders[2] == 0 and orders[3] == 0:
                intor4 = None
            tangent_out += _gen_int2e_jvp_r0(mol, mol_t, [intor1, intor2, intor3, intor4])
        else:
            tangent_out += _int2e_jvp_r0(mol, mol_t, intor1)

    if mol.ctr_coeff is not None:
        tangent_out += _int2e_jvp_cs(mol, mol_t, intor)
    if mol.exp is not None:
        tangent_out += _int2e_jvp_exp(mol, mol_t, intor)
    return primal_out, tangent_out

def _get_fakemol_cs(mol):
    mol1 = uncontract(mol)
    return mol1

def _get_fakemol_exp(mol):
    mol1 = uncontract(mol)
    mol1._bas[:,mole.ANG_OF]  += 2
    return mol1

def get_bas_label(l):
    xyz = []
    for x in range(l, -1, -1):
        for y in range(l-x, -1, -1):
            z = l-x-y
            xyz.append('x'*x + 'y'*y + 'z'*z)
    return xyz

def promote_xyz(xyz, x, l):
    if x == 'z':
        return xyz+'z'*l
    elif x == 'y':
        if 'z' in xyz:
            tmp = xyz.split('z', 1)
            return tmp[0] + 'y'*l + 'z' + tmp[1]
        else:
            return xyz+'y'*l
    elif x == 'x':
        return 'x'*l+xyz
    else:
        raise ValueError

def _int1e_jvp_r0(mol, mol_t, intor):
    s1 = -getints2c(mol, intor, comp=3)
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
        order_a = _int1e_get_dr_order(intor_b)[0]
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
    s1 = -getints2c(mol, intor).reshape(-1,3,nao,nao)
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
    grad = np.zeros((mol.natm,3,nao,nao), dtype=float)
    for k, ia in enumerate(atmlst):
        with mol.with_rinv_at_nucleus(ia):
            vrinv = getints2c_rc(mol, intor, comp=3, rc_deriv=ia)
            if "ECP" not in intor:
                vrinv *= -mol.atom_charge(ia)
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
                order_a = _int1e_get_dr_order(intor_b)[0]
                s1b = getints2c_rc(mol, intor_b, rc_deriv=ia)
                s1b = s1b.reshape(3**order_a,3,-1,nao,nao).transpose(1,0,2,3,4)
                vrinv += s1b.reshape(3,-1,nao,nao)
            if "ECP" not in intor_a:
                vrinv *= -mol.atom_charge(ia)
        grad = ops.index_update(grad, ops.index[k], vrinv)
    jvp = np.einsum('nxyij,nx->yij', grad, coords_t)
    if hermi == 1:
        jvp = jvp + jvp.transpose(0,2,1)
    return jvp

def _int1e_jvp_cs(mol, mol_t, intor):
    ctr_coeff = mol.ctr_coeff
    ctr_coeff_t = mol_t.ctr_coeff

    mol1 = _get_fakemol_cs(mol)
    mol1._atm[:,mole.CHARGE_OF] = 0 # set nuclear charge to zero
    nbas1 = len(mol1._bas)
    nbas = len(mol._bas)
    shls_slice = (0, nbas1, nbas1, nbas1+nbas)
    intor = mol._add_suffix(intor)
    if 'ECP' in intor:
        assert mol._ecp is not None
        bas = numpy.vstack((mol._bas, mol._ecpbas))
    else:
        bas = mol._bas
    atmc, basc, envc = mole.conc_env(mol1._atm, mol1._bas, mol1._env,
                                     mol._atm, bas, mol._env)
    if 'ECP' in intor:
        envc[mole.AS_ECPBAS_OFFSET] = len(mol1._bas) + len(mol._bas)
        envc[mole.AS_NECPBAS] = len(mol._ecpbas)

    s = moleintor.getints(intor, atmc, basc, envc, shls_slice)
    _, cs_of, _ = setup_ctr_coeff(mol)
    nao = mol.nao
    #grad = np.zeros((len(ctr_coeff), nao, nao), dtype=float)
    grad = numpy.zeros((len(ctr_coeff), nao, nao), dtype=float)

    off = 0
    ibas = 0
    for i in range(len(mol._bas)):
        l = mol._bas[i,mole.ANG_OF]
        if mol.cart:
            nbas = (l+1)*(l+2)//2
        else:
            nbas = 2*l + 1
        nprim = mol._bas[i,mole.NPRIM_OF]
        nctr = mol._bas[i,mole.NCTR_OF]
        g = s[off:(off+nprim*nbas)].reshape(nprim,-1,nao)
        for j in range(nctr):
            #idx = ops.index[(cs_of[i]+j*nprim):(cs_of[i]+(j+1)*nprim), ibas:(ibas+nbas), :]
            #grad = ops.index_add(grad, idx, g)
            grad[(cs_of[i]+j*nprim):(cs_of[i]+(j+1)*nprim), ibas:(ibas+nbas)] += g
            ibas += nbas
        off += nprim*nbas
    grad += grad.transpose(0,2,1)

    tangent_out = np.einsum('xij,x->ij', grad, ctr_coeff_t)
    #tangent_out += tangent_out.T
    return tangent_out

def _int1e_jvp_exp(mol, mol_t, intor):
    mol1 = _get_fakemol_exp(mol)
    mol1._atm[:,mole.CHARGE_OF] = 0 # set nuclear charge to zero
    if intor.endswith("_sph"):
        intor = intor.replace("_sph", "_cart")
        cart = False
    else:
        cart = True
        intor = mol._add_suffix(intor, cart=True)

    nbas1 = len(mol1._bas)
    nbas = len(mol._bas)
    shls_slice = (0, nbas1, nbas1, nbas1+nbas)
    if 'ECP' in intor:
        assert mol._ecp is not None
        bas = numpy.vstack((mol._bas, mol._ecpbas))
    else:
        bas = mol._bas
    atmc, basc, envc = mole.conc_env(mol1._atm, mol1._bas, mol1._env,
                                     mol._atm, bas, mol._env)
    if 'ECP' in intor:
        envc[mole.AS_ECPBAS_OFFSET] = len(mol1._bas) + len(mol._bas)
        envc[mole.AS_NECPBAS] = len(mol._ecpbas)

    s = moleintor.getints(intor, atmc, basc, envc, shls_slice)
    es, es_of, _env_of = setup_exp(mol)
    nao = mole.nao_cart(mol)
    #grad = np.zeros((len(es), nao, nao), dtype=float)
    grad = numpy.zeros((len(es), nao, nao), dtype=float)

    off = 0
    ibas = 0
    for i in range(len(mol._bas)):
        ioff = es_of[i]

        l = mol._bas[i,mole.ANG_OF]
        nbas = (l+1)*(l+2)//2
        nbas1 = (l+3)*(l+4)//2
        nprim = mol._bas[i,mole.NPRIM_OF]
        nctr = mol._bas[i,mole.NCTR_OF]
        ptr_ctr_coeff = mol._bas[i,mole.PTR_COEFF]
        g = s[off:off+nprim*nbas1].reshape(nprim, nbas1, nao)

        xyz = get_bas_label(l)
        xyz1 = get_bas_label(l+2)
        for k in range(nctr):
            for j in range(nprim):
                c = mol._env[ptr_ctr_coeff + k*nprim + j]
                if l == 0:
                    c *= 0.282094791773878143 # normalization factor for s orbital
                elif l == 1:
                    c *= 0.488602511902919921 # normalization factor for p orbital
                jbas = ibas
                for orb in xyz:
                    idx_x = xyz1.index(promote_xyz(orb, 'x', 2))
                    idx_y = xyz1.index(promote_xyz(orb, 'y', 2))
                    idx_z = xyz1.index(promote_xyz(orb, 'z', 2))
                    gc = -(g[j,idx_x] + g[j,idx_y] + g[j,idx_z]) *  c
                    #grad = ops.index_add(grad, ops.index[ioff+j, jbas], gc)
                    grad[ioff+j, jbas] += gc
                    jbas += 1
            ibas += nbas
        off += nprim * nbas1

    grad += grad.transpose(0,2,1)
    tangent_out = np.einsum('xij,x->ij', grad, mol_t.exp)
    #tangent_out += tangent_out.T
    if not mol.cart or not cart:
        c2s = np.asarray(mol.cart2sph_coeff())
        tangent_out = np.dot(c2s.T, np.dot(tangent_out, c2s))
    return tangent_out

def _int2e_jvp_r0(mol, mol_t, intor):
    coords_t = mol_t.coords
    #atmlst = range(mol.natm)
    #aoslices = numpy.asarray(mol.aoslice_by_atom(), dtype=numpy.int32)
    #nao = mol.nao

    #eri1 = -Mole.intor(mol, intor, comp=3, aosym='s2kl')
    #grad = numpy.zeros((mol.natm,3,nao,nao,nao,nao), dtype=numpy.double)
    #libcgto.restore_int2e_deriv(grad.ctypes.data_as(ctypes.c_void_p),
    #        eri1.ctypes.data_as(ctypes.c_void_p),
    #        aoslices.ctypes.data_as(ctypes.c_void_p),
    #        ctypes.c_int(mol.natm), ctypes.c_int(nao))
    #eri1 = None
    eri1 = -getints4c(mol, intor, comp=None, aosym='s1')
    aoslices = mol.aoslice_by_atom()[:,2:4]
    grad = _fill_grad_r0(eri1, aoslices)

    #for k, ia in enumerate(atmlst):
    #    p0, p1 = aoslices [ia,2:]
    #    tmp = np.einsum("xijkl,x->ijkl", eri1[:,p0:p1], coords_t[k])
    #    tangent_out = ops.index_add(tangent_out, ops.index[p0:p1], tmp)
    #    tangent_out = ops.index_add(tangent_out, ops.index[:,p0:p1], tmp.transpose((1,0,2,3)))
    #    tangent_out = ops.index_add(tangent_out, ops.index[:,:,p0:p1], tmp.transpose((2,3,0,1)))
    #    tangent_out = ops.index_add(tangent_out, ops.index[:,:,:,p0:p1], tmp.transpose((2,3,1,0)))
    return _int2e_dot_grad_tangent_r0(grad, coords_t)

@jit
def _int2e_dot_grad_tangent_r0(grad, tangent):
    tangent_out = np.einsum("nxijkl,nx->ijkl", grad, tangent)
    tangent_out += tangent_out.transpose(1,0,2,3)
    tangent_out += tangent_out.transpose(2,3,0,1)
    return tangent_out

def _gen_int2e_jvp_r0(mol, mol_t, intors):
    coords_t = mol_t.coords
    nao = mol.nao
    intor_a, intor_b, intor_c, intor_d = intors
    eri1_a = -getints4c(mol, intor_a, aosym='s1').reshape(3,-1,nao,nao,nao,nao)
    if intor_b:
        orders = _int2e_get_dr_order(intor_b)
        off = 3**orders[0]
        eri1_b = -getints4c(mol, intor_b, aosym='s1')
        eri1_b = eri1_b.reshape(off,3,-1,nao,nao,nao,nao)
        eri1_b = eri1_b.transpose(1,0,2,3,4,5,6).reshape(3,-1,nao,nao,nao,nao)
    else:
        eri1_b = eri1_a.transpose(0,1,3,2,4,5)

    orders = _int2e_get_dr_order(intor_c)
    off = 3**(orders[0]+orders[1])
    eri1_c = -getints4c(mol, intor_c, aosym='s1')
    eri1_c = eri1_c.reshape(off,3,-1,nao,nao,nao,nao)
    eri1_c = eri1_c.transpose(1,0,2,3,4,5,6).reshape(3,-1,nao,nao,nao,nao)
    if intor_d:
        orders = _int2e_get_dr_order(intor_d)
        off = 3**(orders[0]+orders[1]+orders[2])
        eri1_d = -getints4c(mol, intor_d, aosym='s1')
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
    jvp = np.einsum("nxyijkl,nx->yijkl", grad, coords_t)
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

def _int2e_jvp_cs(mol, mol_t, intor):
    ctr_coeff = mol.ctr_coeff
    ctr_coeff_t = mol_t.ctr_coeff

    mol1 = _get_fakemol_cs(mol)

    nbas = len(mol._bas)
    nbas1 = len(mol1._bas)
    atmc, basc, envc = mole.conc_env(mol1._atm, mol1._bas, mol1._env,
                                     mol._atm, mol._bas, mol._env)

    shls_slice = (0, nbas1, nbas1, nbas1+nbas, nbas1, nbas1+nbas, nbas1, nbas1+nbas)
    intor = mol._add_suffix(intor)
    eri = moleintor.getints(intor, atmc, basc, envc, shls_slice)

    _, cs_of, _ = setup_ctr_coeff(mol)
    nao = mol.nao
    #grad = np.zeros((len(ctr_coeff), nao, nao, nao, nao), dtype=float)
    grad = numpy.zeros((len(ctr_coeff), nao, nao, nao, nao), dtype=eri.dtype)

    off = 0
    ibas = 0
    for i in range(len(mol._bas)):
        l = mol._bas[i,mole.ANG_OF]
        if mol.cart:
            nbas = (l+1)*(l+2)//2
        else:
            nbas = 2*l + 1
        nprim = mol._bas[i,mole.NPRIM_OF]
        nctr = mol._bas[i,mole.NCTR_OF]
        g = eri[off:(off+nprim*nbas)].reshape(nprim,-1,nao,nao,nao)
        for j in range(nctr):
            #idx = ops.index[(cs_of[i]+j*nprim):(cs_of[i]+(j+1)*nprim), ibas:(ibas+nbas)]
            #grad = ops.index_add(grad, idx, g)
            grad[(cs_of[i]+j*nprim):(cs_of[i]+(j+1)*nprim), ibas:(ibas+nbas)] += g
            ibas += nbas
        off += nprim*nbas
    #grad += grad.transpose(0,2,1,3,4)
    #grad += grad.transpose(0,3,4,1,2)
    tangent_out = _int2e_dot_grad_tangent_cs(grad, ctr_coeff_t)
    return tangent_out

@jit
def _int2e_dot_grad_tangent_cs(grad, tangent):
    tangent_out = np.einsum('xijkl,x->ijkl', grad, tangent)
    tangent_out += tangent_out.transpose(1,0,2,3)
    tangent_out += tangent_out.transpose(2,3,0,1)
    return tangent_out

def _int2e_jvp_exp(mol, mol_t, intor):
    mol1 = _get_fakemol_exp(mol)
    intor = mol._add_suffix(intor, cart=True)

    nbas = len(mol._bas)
    nbas1 = len(mol1._bas)
    atmc, basc, envc = mole.conc_env(mol1._atm, mol1._bas, mol1._env,
                                     mol._atm, mol._bas, mol._env)
    shls_slice = (0, nbas1, nbas1, nbas1+nbas, nbas1, nbas1+nbas, nbas1, nbas1+nbas)

    eri = moleintor.getints(intor, atmc, basc, envc, shls_slice)

    es, es_of, _env_of = setup_exp(mol)
    nao = mole.nao_cart(mol)
    #grad = np.zeros((len(es), nao, nao, nao, nao), dtype=float)
    grad = numpy.zeros((len(es), nao, nao, nao, nao), dtype=float)

    off = 0
    ibas = 0
    for i in range(len(mol._bas)):
        ioff = es_of[i]

        l = mol._bas[i,mole.ANG_OF]
        nbas = (l+1)*(l+2)//2
        nbas1 = (l+3)*(l+4)//2
        nprim = mol._bas[i,mole.NPRIM_OF]
        nctr = mol._bas[i,mole.NCTR_OF]
        ptr_ctr_coeff = mol._bas[i,mole.PTR_COEFF]
        g = eri[off:off+nprim*nbas1].reshape(nprim, nbas1, nao, nao, nao)

        xyz = get_bas_label(l)
        xyz1 = get_bas_label(l+2)
        for k in range(nctr):
            for j in range(nprim):
                c = mol._env[ptr_ctr_coeff + k*nprim + j]
                if l == 0:
                    c *= 0.282094791773878143 # normalization factor for s orbital
                elif l == 1:
                    c *= 0.488602511902919921 # normalization factor for p orbital
                jbas = ibas
                for orb in xyz:
                    idx_x = xyz1.index(promote_xyz(orb, 'x', 2))
                    idx_y = xyz1.index(promote_xyz(orb, 'y', 2))
                    idx_z = xyz1.index(promote_xyz(orb, 'z', 2))
                    gc = -(g[j,idx_x] + g[j,idx_y] + g[j,idx_z]) *  c
                    #grad = ops.index_add(grad, ops.index[ioff+j, jbas], gc)
                    grad[ioff+j, jbas] += gc
                    jbas += 1
            ibas += nbas
        off += nprim * nbas1

    #grad += grad.transpose(0,2,1,3,4)
    #grad += grad.transpose(0,3,4,1,2)
    tangent_out = _int2e_dot_grad_tangent_exp(grad, mol_t.exp)
    if not mol.cart:
        c2s = numpy.asarray(mol.cart2sph_coeff())
        tangent_out = _int2e_c2s(tangent_out, c2s)
    return tangent_out

@jit
def _int2e_dot_grad_tangent_exp(grad, tangent):
    tangent_out = np.einsum('xijkl,x->ijkl', grad, tangent)
    tangent_out += tangent_out.transpose(1,0,2,3)
    tangent_out += tangent_out.transpose(2,3,0,1)
    return tangent_out

@jit
def _int2e_c2s(eris_cart, c2s):
    eris_sph = np.einsum("iu,jv,ijkl,ks,lt->uvst", c2s, c2s, eris_cart, c2s, c2s)
    return eris_sph
