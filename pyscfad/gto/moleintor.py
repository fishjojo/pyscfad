from functools import partial
import ctypes
import numpy
from jax import vmap
from jax import custom_jvp
from jax import jit
from jax.lax import dynamic_slice, dynamic_update_slice
from pyscf import ao2mo
from pyscf.gto import mole, moleintor
from pyscf.gto.mole import Mole
from pyscf.gto.moleintor import _get_intor_and_comp
from pyscfad.lib import numpy as np
from pyscfad.lib import ops
from pyscfad.lib.misc import load_library
from ._mole_helper import uncontract, setup_exp, setup_ctr_coeff

libcgto = load_library('libcgto')

SET_RC = ["rinv",]

def getints(mol, intor, shls_slice=None,
            comp=None, hermi=0, aosym='s1', out=None):
    if intor.endswith("_spinor"):
        raise NotImplementedError

    if (intor.startswith("int1e") or
        intor.startswith("int2c2e") or
        intor.startswith('ECP')):
        return getints2c(mol, intor, shls_slice, comp, hermi, aosym, out)
    elif intor.startswith("int2e"):
        return getints4c(mol, intor, shls_slice, comp, aosym, out)
    else:
        raise NotImplementedError

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
        intor_ip_bra = intor.replace("int1e_", "int1e_ip")
        if "sph" in intor:
            intor_ip_ket = intor.replace("_sph","") + "ip" + "_sph"
        elif "cart" in intor:
            intor_ip_ket = intor.replace("_cart","") + "ip" + "_cart"
        else:
            intor_ip_ket = intor + "ip"
        tangent_out += _gen_int1e_jvp_r0(mol, mol_t, intor_ip_bra, intor_ip_ket, rc_deriv)
    return primal_out, tangent_out

@partial(custom_jvp, nondiff_argnums=tuple(range(1,7)))
def getints2c(mol, intor,
              shls_slice=None, comp=None, hermi=0, aosym='s1', out=None):
    return Mole.intor(mol, intor,
                      comp=comp, hermi=hermi, aosym=aosym,
                      shls_slice=shls_slice, out=out)

@getints2c.defjvp
def getints2c_jvp(intor, shls_slice, comp, hermi, aosym, out,
                  primals, tangents):
    if shls_slice is not None:
        raise NotImplementedError

    mol, = primals
    mol_t, = tangents

    primal_out = getints2c(mol, intor, shls_slice=None,
                           comp=comp, hermi=hermi, aosym=aosym, out=out)

    tangent_out = np.zeros_like(primal_out)
    if mol.coords is not None:
        intor_ip_bra = intor_ip_ket = intor_ip = None
        if intor.startswith("ECPscalar"):
            intor_ip = intor.replace("ECPscalar", "ECPscalar_ipnuc")
        elif intor.startswith("int2c2e"):
            intor_ip = intor.replace("int2c2e", "int2c2e_ip1")
        elif intor.startswith("int1e"):
            if "ip" in intor:
                intor_ip_bra = intor.replace("int1e_", "int1e_ip")
                if "sph" in intor:
                    intor_ip_ket = intor.replace("_sph","") + "ip" + "_sph"
                elif "cart" in intor:
                    intor_ip_ket = intor.replace("_cart","") + "ip" + "_cart"
                else:
                    intor_ip_ket = intor + "ip"
            elif intor in ['int1e_r', 'int1e_r_sph', 'int1e_r_cart']:
                intor_ip = intor.replace('int1e_r', 'int1e_irp')
            else:
                intor_ip = intor.replace("int1e_", "int1e_ip")
        if intor_ip_bra or intor_ip_ket:
            tangent_out += _gen_int1e_jvp_r0(mol, mol_t, intor_ip_bra, intor_ip_ket)
            if "nuc" in intor_ip_bra and "nuc" in intor_ip_ket:
                intor_ip_bra = intor_ip_bra.replace("nuc", "rinv")
                intor_ip_ket = intor_ip_ket.replace("nuc", "rinv")
                tangent_out += _gen_int1e_nuc_jvp_rc(mol, mol_t, intor_ip_bra, intor_ip_ket)
            return primal_out, tangent_out
        elif intor_ip.startswith("int1e_ir"):
            tangent_out += _int1e_r_jvp_r0(mol, mol_t, intor_ip)
        else:
            tangent_out += _int1e_jvp_r0(mol, mol_t, intor_ip)

        intor_ip = None
        if intor.startswith("int1e_nuc"):
            intor_ip = intor.replace("int1e_nuc", "int1e_iprinv")
        elif intor.startswith("ECPscalar"):
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
        eri8 = None
    else:
        eri = Mole.intor(mol, intor,
                comp=comp, aosym=aosym,
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

    if mol.coords is not None:
        if "ip" in intor:
            str12 = intor.replace("int2e_","").replace("_sph","").replace("_cart","")
            if "1" in str12:
                str1 = str12[:str12.index("1")+1]
                str_a = "ip" + str12
                if not "v" in str1:
                    str_b = str12.replace("1", "vip1")
                else:
                    str_b = str12.replace("1", "ip1")
            else:
                str_a = "ip1" + str12
                str_b = None
            if "2" in str12:
                if "1" in str12:
                    str_c = str12.replace("1", "1ip")
                    str2 = str12[str12.index("1")+1:]
                else:
                    str_c = "ip" + str12
                    str2 = str12
                if not "v" in str2:
                    str_d = str12.replace("2", "vip2")
                else:
                    str_d = str12.replace("2", "ip2")
            else:
                str_c = str12 + "ip2"
                str_d = None

            if "sph" in intor:
                suffix = "_sph"
            elif "cart" in intor:
                suffix = "_cart"
            else:
                suffix = ""
            intors = [str_a, str_b, str_c, str_d]
            for i, intor_i in enumerate(intors):
                if intor_i:
                    intors[i] = "int2e_" + intor_i + suffix
            tangent_out += _gen_int2e_jvp_r0(mol, mol_t, intors)
        else:
            intor_ip = intor.replace("int2e", "int2e_ip1")
            tangent_out += _int2e_jvp_r0(mol, mol_t, intor_ip)
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
    grad = _int1e_fill_grad_r0(mol, s1)
    tangent_out = _int1e_dot_grad_tangent_r0(grad, mol_t.coords)
    return tangent_out

# pylint: disable=fixme
# TODO unrolling the for loop can be slow,
# and vmap does not work with dynamically shaped sub-array.
# How to make in-place assignment efficient?
# jit breaks pytest
#@jit
def _int1e_fill_grad_r0(mol, s1):
    shape = [mol.natm,] + list(s1.shape)
    grad = np.zeros(shape, dtype=s1.dtype)
    aoslices = mol.aoslice_by_atom()
    for ia in range(mol.natm):
        p0, p1 = aoslices[ia,2:]
        grad = ops.index_update(grad, ops.index[ia,:,p0:p1], s1[:,p0:p1])
    #def func(grad_k, ia):
    #    p0, p1 = aoslices[ia, 2:]
    #    return ops.index_update(grad_k, ops.index[:,p0:p1], s1[:,p0:p1])
    #grad = vmap(func)(grad, np.arange(mol.natm))
    return grad

def _gen_int1e_jvp_r0(mol, mol_t, intor_a, intor_b, rc_deriv=None):
    coords_t = mol_t.coords
    #atmlst = range(mol.natm)
    #aoslices = mol.aoslice_by_atom()
    nao = mol.nao

    s1a = -getints2c(mol, intor_a).reshape(3,-1,nao,nao)
    s1b = -getints2c(mol, intor_b).reshape(-1,3,nao,nao).transpose(1,0,2,3)
    #jvp = np.zeros(s1a.shape[1:])
    #for k, ia in enumerate(atmlst):
    #    p0, p1 = aoslices[ia,2:]
    #    ta = np.einsum('xyij,x->yij', s1a[...,p0:p1,:], coords_t[k])
    #    tb = np.einsum('xyij,x->yij', s1b[...,p0:p1], coords_t[k])
    #    jvp = ops.index_add(jvp, ops.index[:,p0:p1], ta)
    #    jvp = ops.index_add(jvp, ops.index[:,:,p0:p1], tb)
    grad = _gen_int1e_fill_grad_r0(mol, s1a, s1b)
    if rc_deriv is not None:
        grad = ops.index_add(grad, ops.index[rc_deriv], -s1a-s1b)
    jvp = _gen_int1e_dot_grad_tangent_r0(grad, coords_t)
    return jvp

@jit
def _gen_int1e_fill_grad_r0(mol, s1a, s1b):
    shape = [mol.natm,] + list(s1a.shape)
    grad = np.zeros(shape, dtype=s1a.dtype)
    aoslices = mol.aoslice_by_atom()
    for ia in range(mol.natm):
        p0, p1 = aoslices[ia,2:]
        grad = ops.index_add(grad, ops.index[ia,:,:,p0:p1], s1a[...,p0:p1,:])
        grad = ops.index_add(grad, ops.index[ia,...,p0:p1], s1b[...,p0:p1])
    return grad

@jit
def _gen_int1e_dot_grad_tangent_r0(grad, tangent):
    tangent_out = np.einsum('nxyij,nx->yij', grad, tangent)
    return tangent_out

def _int1e_r_jvp_r0(mol, mol_t, intor):
    coords_t = mol_t.coords
    atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()
    nao = mol.nao
    s1 = -Mole.intor(mol, intor).reshape(3,-1,nao,nao)
    grad = [numpy.zeros_like(s1) for ia in atmlst]
    grad = numpy.asarray(grad)
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices [ia,2:]
        grad[k,...,p0:p1] = s1[...,p0:p1]
    tangent_out = _int1e_r_dot_grad_tangent_r0(grad, coords_t)
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

def _gen_int1e_nuc_jvp_rc(mol, mol_t, intor_a, intor_b):
    coords_t = mol_t.coords
    atmlst = range(mol.natm)
    nao = mol.nao
    _, comp = _get_intor_and_comp(intor_a)
    grad = np.zeros((mol.natm,3,comp//3,nao,nao), dtype=float)
    for k, ia in enumerate(atmlst):
        with mol.with_rinv_at_nucleus(ia):
            vrinv = getints2c_rc(mol, intor_a, rc_deriv=ia).reshape(3,-1,nao,nao)
            vrinv+= getints2c_rc(mol, intor_b, rc_deriv=ia).reshape(-1,3,nao,nao).transpose(1,0,2,3)
            if "ECP" not in intor_a:
                vrinv *= -mol.atom_charge(ia)
        grad = ops.index_update(grad, ops.index[k], vrinv)
    tangent_out = _gen_int1e_dot_grad_tangent_r0(grad, coords_t)
    return tangent_out


@jit
def _int1e_dot_grad_tangent_r0(grad, tangent):
    tangent_out = np.einsum('nxij,nx->ij', grad, tangent)
    tangent_out += tangent_out.T
    return tangent_out

@jit
def _int1e_r_dot_grad_tangent_r0(grad, tangent):
    tangent_out = np.einsum('nxpij,nx->pij', grad, tangent)
    tangent_out += tangent_out.transpose(0,2,1)
    return tangent_out

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
    grad = _int2e_fill_grad_r0(mol, eri1)
    #for k, ia in enumerate(atmlst):
    #    p0, p1 = aoslices [ia,2:]
    #    tmp = np.einsum("xijkl,x->ijkl", eri1[:,p0:p1], coords_t[k])
    #    tangent_out = ops.index_add(tangent_out, ops.index[p0:p1], tmp)
    #    tangent_out = ops.index_add(tangent_out, ops.index[:,p0:p1], tmp.transpose((1,0,2,3)))
    #    tangent_out = ops.index_add(tangent_out, ops.index[:,:,p0:p1], tmp.transpose((2,3,0,1)))
    #    tangent_out = ops.index_add(tangent_out, ops.index[:,:,:,p0:p1], tmp.transpose((2,3,1,0)))
    return _int2e_dot_grad_tangent_r0(grad, coords_t)

#@jit
def _int2e_fill_grad_r0(mol, eri1):
    shape = [mol.natm,] + list(eri1.shape)
    grad = np.zeros(shape, dtype=eri1.dtype)
    aoslices = mol.aoslice_by_atom()
    for ia in range(mol.natm):
        p0, p1 = aoslices[ia,2:]
        grad = ops.index_update(grad, ops.index[ia,:,p0:p1], eri1[:,p0:p1])
    return grad

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
    # pylint: disable=fixme
    # FIXME the shapes of eris are wrong for 3rd and higher order derivatives
    eri1_a = -getints4c(mol, intor_a, aosym='s1').reshape(3,-1,nao,nao,nao,nao)
    if intor_b:
        eri1_b = -getints4c(mol, intor_b, aosym='s1')
        eri1_b = eri1_b.reshape(-1,3,nao,nao,nao,nao).transpose(1,0,2,3,4,5)
    else:
        eri1_b = eri1_a.transpose(0,1,3,2,4,5)
    eri1_c = -getints4c(mol, intor_c, aosym='s1')
    eri1_c = eri1_c.reshape(-1,3,nao,nao,nao,nao).transpose(1,0,2,3,4,5)
    if intor_d:
        eri1_d = -getints4c(mol, intor_d, aosym='s1')
        eri1_d = eri1_d.reshape(-1,3,nao,nao,nao,nao).transpose(1,0,2,3,4,5)
    else:
        eri1_d = eri1_c.transpose(0,1,2,3,5,4)
    grad = _gen_int2e_fill_grad_r0(mol, eri1_a, eri1_b, eri1_c, eri1_d)
    return _gen_int2e_dot_grad_tangent_r0(grad, coords_t)

@jit
def _gen_int2e_fill_grad_r0(mol, eri1_a, eri1_b, eri1_c, eri1_d):
    shape = [mol.natm,] + list(eri1_a.shape)
    grad = np.zeros(shape, dtype=eri1_a.dtype)
    aoslices = mol.aoslice_by_atom()
    for ia in range(mol.natm):
        p0, p1 = aoslices[ia,2:]
        grad = ops.index_add(grad, ops.index[ia,:,:,p0:p1], eri1_a[:,:,p0:p1])
        grad = ops.index_add(grad, ops.index[ia,:,:,:,p0:p1], eri1_b[:,:,:,p0:p1])
        grad = ops.index_add(grad, ops.index[ia,...,p0:p1,:], eri1_c[...,p0:p1,:])
        grad = ops.index_add(grad, ops.index[ia,...,p0:p1], eri1_d[...,p0:p1])
    return grad

@jit
def _gen_int2e_dot_grad_tangent_r0(grad, tangent):
    tangent_out = np.einsum("nxyijkl,nx->yijkl", grad, tangent)
    return tangent_out

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
    grad = numpy.zeros((len(ctr_coeff), nao, nao, nao, nao), dtype=numpy.double)

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
