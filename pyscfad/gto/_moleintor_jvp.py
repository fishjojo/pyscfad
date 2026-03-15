# Copyright 2021-2025 The PySCFAD Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
import numpy

from pyscf import ao2mo
from pyscf.gto import mole as pyscf_mole
from pyscf.gto import ATOM_OF
from pyscf.gto.moleintor import _get_intor_and_comp

from pyscfad import numpy as np
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
        tangent_out += _gen_int1e_fill_jvp_r0(
                            s1a, mol1_t.coords, aoslices1,
                            idx1[None,None,:,None]).reshape(primal_out.shape)

    if mol2.coords is not None:
        aoslices2 = mol2.aoslice_by_atom()[:,2:4]
        _, intor_ip_ket = int1e_dr1_name(intor)
        s1b = -intor_cross(intor_ip_ket, mol1, mol2, comp=None, grids=grids)

        order_a = int1e_get_dr_order(intor_ip_ket)[0]
        s1b = s1b.reshape(3**order_a,3,-1,nao1,nao2).transpose(1,0,2,3,4).reshape(3,-1,nao1,nao2)
        idx2 = np.arange(nao2)
        tangent_out += _gen_int1e_fill_jvp_r0(
                            s1b, mol2_t.coords, aoslices2,
                            idx2[None,None,None,:]).reshape(primal_out.shape)

    if mol1.ctr_coeff is not None:
        raise NotImplementedError
    if mol1.exp is not None:
        raise NotImplementedError

    if mol2.ctr_coeff is not None:
        raise NotImplementedError
    if mol2.exp is not None:
        raise NotImplementedError

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
    if hermi == 2:
        raise NotImplementedError

    mol, = primals
    mol_t, = tangents
    primal_out = _getints2c_rc(mol, intor, shls_slice, comp, hermi, out, rc_deriv)
    tangent_out = np.zeros_like(primal_out)

    if mol.coords is not None:
        intor_ip_bra, intor_ip_ket = int1e_dr1_name(intor)
        tangent_out += _gen_int1e_jvp_r0(mol, mol_t, intor_ip_bra, intor_ip_ket,
                                         rc_deriv=rc_deriv, shls_slice=shls_slice)

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
            tangent_out += _gen_int1e_jvp_r0(
                                mol, mol_t, intor_ip_bra, intor_ip_ket,
                                hermi=hermi, shls_slice=shls_slice).reshape(tangent_out.shape)
            if 'nuc' in intor_ip_bra and 'nuc' in intor_ip_ket:
                intor_ip_bra = intor_ip_bra.replace('nuc', 'rinv')
                intor_ip_ket = intor_ip_ket.replace('nuc', 'rinv')
                tangent_out += _gen_int1e_nuc_jvp_rc(
                                mol, mol_t, intor_ip_bra, intor_ip_ket,
                                hermi=hermi, shls_slice=shls_slice).reshape(tangent_out.shape)
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
            tangent_out += _gen_int2e_jvp_r0(
                                mol, mol_t,
                                (intor1, intor2, intor3, intor4),
                                shls_slice=shls_slice)
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
    idx = np.arange(mol.nao)[None,:,None]
    jvp = _int1e_fill_jvp_r0_s2(s1, mol_t.coords, aoslices, idx)
    return jvp

@jit
def _int1e_fill_jvp_r0_s2(ints, coords_t, aoslices, aoidx):
    def _fill(sl, coord_t):
        mask = (aoidx >= sl[0]) & (aoidx < sl[1])
        grad = np.where(mask, ints, np.array(0, dtype=ints.dtype))
        return np.einsum('xij,x->ij', grad, coord_t)
    jvp = np.sum(vmap(_fill)(aoslices, coords_t), axis=0)
    jvp += jvp.T.conj()
    return jvp

def _gen_int1e_jvp_r0(mol, mol_t, intor_a, intor_b,
                      rc_deriv=None, hermi=0, shls_slice=None):
    if shls_slice is None:
        shls_slice = (0, mol.nbas, 0, mol.nbas)
    ao_loc = moleintor.make_loc(mol._bas, intor_a)
    i0, i1, j0, j1 = shls_slice[:4]
    if hermi == 1:
        assert (i0 == j0 and i1 == j1)
    naoi = ao_loc[i1] - ao_loc[i0]
    naoj = ao_loc[j1] - ao_loc[j0]

    s1a = -getints2c_rc(mol, intor_a,
                        hermi=0, rc_deriv=rc_deriv,
                        shls_slice=shls_slice).reshape(3,-1,naoi,naoj)

    coords_t = mol_t.coords
    aoslices = mol.aoslice_by_atom()[:,2:4] - ao_loc[i0]
    aoidx = np.arange(naoi)
    jvp = _gen_int1e_fill_jvp_r0(s1a, coords_t, aoslices, aoidx[None,None,:,None])
    if rc_deriv is not None:
        jvp += np.einsum('xyij,x->yij', -s1a, coords_t[rc_deriv])

    if hermi == 0:
        order_a = int1e_get_dr_order(intor_b)[0]
        s1b = -getints2c_rc(mol, intor_b, hermi=0, rc_deriv=rc_deriv,
                            shls_slice=shls_slice)
        s1b = s1b.reshape(3**order_a,3,-1,naoi,naoj).transpose(1,0,2,3,4).reshape(3,-1,naoi,naoj)

        aoslices = mol.aoslice_by_atom()[:,2:4] - ao_loc[j0]
        aoidx = np.arange(naoj)
        jvp += _gen_int1e_fill_jvp_r0(s1b, coords_t, aoslices, aoidx[None,None,None,:])
        if rc_deriv is not None:
            jvp += np.einsum('xyij,x->yij', -s1b, coords_t[rc_deriv])
    elif hermi == 1:
        jvp += jvp.transpose(0,2,1)
    return jvp

@jit
def _gen_int1e_fill_jvp_r0(ints, coords_t, aoslices, aoidx):
    def _fill(sl, coord_t):
        mask = (aoidx >= sl[0]) & (aoidx < sl[1])
        grad = np.where(mask, ints, np.array(0, dtype=ints.dtype))
        return np.einsum('xyij,x->yij', grad, coord_t)
    jvp = np.sum(vmap(_fill)(aoslices, coords_t), axis=0)
    return jvp

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

    jvp = np.zeros((nao,nao), dtype=float)
    ecp_intor = 'ECP' in intor
    if ecp_intor:
        if not mol.has_ecp():
            return jvp
        else:
            ecp_atoms = set(mol._ecpbas[:,ATOM_OF])

    for k, ia in enumerate(atmlst):
        with mol.with_rinv_at_nucleus(ia):
            if not ecp_intor:
                vrinv = getints2c_rc(mol, intor, comp=3, rc_deriv=ia)
                vrinv *= -mol.atom_charge(ia)
            else:
                if ia in ecp_atoms:
                    vrinv = getints2c_rc(mol, intor, comp=3, rc_deriv=ia)
                else:
                    continue
        jvp_k = np.einsum('xij,x->ij', vrinv, coords_t[ia])
        jvp_k += jvp_k.T
        jvp += jvp_k
    return jvp

def _gen_int1e_nuc_jvp_rc(mol, mol_t, intor_a, intor_b,
                          hermi=0, shls_slice=None):
    if shls_slice is None:
        shls_slice = (0, mol.nbas, 0, mol.nbas)
    ao_loc = moleintor.make_loc(mol._bas, intor_a)
    i0, i1, j0, j1 = shls_slice[:4]
    if hermi == 1:
        assert (i0 == j0 and i1 == j1)
    naoi = ao_loc[i1] - ao_loc[i0]
    naoj = ao_loc[j1] - ao_loc[j0]

    atmlst = range(mol.natm)
    _, comp = _get_intor_and_comp(intor_a)
    jvp = np.zeros((comp//3,naoi,naoj), dtype=float)
    for k, ia in enumerate(atmlst):
        with mol.with_rinv_at_nucleus(ia):
            vrinv = getints2c_rc(mol, intor_a, rc_deriv=ia,
                                 shls_slice=shls_slice).reshape(3,-1,naoi,naoj)
            if hermi == 0:
                order_a = int1e_get_dr_order(intor_b)[0]
                s1b = getints2c_rc(mol, intor_b, rc_deriv=ia, shls_slice=shls_slice)
                s1b = s1b.reshape(3**order_a,3,-1,naoi,naoj).transpose(1,0,2,3,4)
                vrinv += s1b.reshape(3,-1,naoi,naoj)
            if 'ECP' not in intor_a:
                vrinv *= -mol.atom_charge(ia)
        jvp += np.einsum('xyij,x->yij', vrinv, mol_t.coords[ia])
    if hermi == 1:
        jvp = jvp + jvp.transpose(0,2,1)
    return jvp

def _int1e_jvp_cs(mol, mol_t, intor, shls_slice, comp, hermi):
    nbas = mol.nbas
    if shls_slice is not None:
        assert shls_slice == (0, nbas, 0, nbas)

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
    nbas = mol.nbas
    if shls_slice is not None:
        assert shls_slice == (0, nbas, 0, nbas)

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
    idx = np.arange(mol.nao)[:,None,None,None]
    return _int2e_fill_jvp_r0_s4(eri1, mol_t.coords, aoslices, idx)

@jit
def _int2e_fill_jvp_r0_s4(ints, coords_t, aoslices, aoidx):
    def _fill(sl, coord_t):
        mask = (aoidx >= sl[0]) & (aoidx < sl[1])
        grad = np.where(mask, ints, np.array(0, dtype=ints.dtype))
        return np.einsum('xijkl,x->ijkl', grad, coord_t)
    jvp = np.sum(vmap(_fill)(aoslices, coords_t), axis=0)
    jvp += jvp.transpose(1,0,2,3)
    jvp += jvp.transpose(2,3,0,1)
    return jvp

def _gen_int2e_jvp_r0(mol, mol_t, intors, shls_slice=None):
    nao = mol.nao
    intor_a, intor_b, intor_c, intor_d = intors

    nbas = mol.nbas
    if shls_slice is None:
        shls_slice = (0, nbas, 0, nbas, 0, nbas, 0, nbas)
    ao_loc = moleintor.make_loc(mol._bas, intor_a)
    i0, i1, j0, j1, k0, k1, l0, l1 = shls_slice[:]
    naoi = ao_loc[i1] - ao_loc[i0]
    naoj = ao_loc[j1] - ao_loc[j0]
    naok = ao_loc[k1] - ao_loc[k0]
    naol = ao_loc[l1] - ao_loc[l0]

    eri1_a = -intor4c(mol, intor_a, aosym='s1',
                      shls_slice=shls_slice).reshape(3,-1,naoi,naoj,naok,naol)

    coords_t = mol_t.coords
    aoslices = mol.aoslice_by_atom()[:,2:4]
    aoslices_a = aoslices - ao_loc[i0]
    idx_a = np.arange(naoi)[None,None,:,None,None,None]
    jvp = _gen_int2e_fill_jvp_r0(eri1_a, coords_t, aoslices_a, idx_a)

    if intor_b:
        orders = int2e_get_dr_order(intor_b)
        off = 3**orders[0]
        eri1_b = -intor4c(mol, intor_b, aosym='s1', shls_slice=shls_slice)
        eri1_b = eri1_b.reshape(off,3,-1,naoi,naoj,naok,naol)
        eri1_b = eri1_b.transpose(1,0,2,3,4,5,6).reshape(3,-1,naoi,naoj,naok,naol)

        aoslices_b = aoslices - ao_loc[j0]
        idx_b = np.arange(naoj)[None,None,None,:,None,None]
        jvp += _gen_int2e_fill_jvp_r0(eri1_b, coords_t, aoslices_b, idx_b)
    else:
        jvp += jvp.transpose(0,2,1,3,4)

    orders = int2e_get_dr_order(intor_c)
    off = 3**(orders[0]+orders[1])
    eri1_c = -intor4c(mol, intor_c, aosym='s1', shls_slice=shls_slice)
    eri1_c = eri1_c.reshape(off,3,-1,naoi,naoj,naok,naol)
    eri1_c = eri1_c.transpose(1,0,2,3,4,5,6).reshape(3,-1,naoi,naoj,naok,naol)

    aoslices_c = aoslices - ao_loc[k0]
    idx_c = np.arange(naok)[None,None,None,None,:,None]
    jvp_c = _gen_int2e_fill_jvp_r0(eri1_c, coords_t, aoslices_c, idx_c)
    jvp += jvp_c

    if intor_d:
        orders = int2e_get_dr_order(intor_d)
        off = 3**(orders[0]+orders[1]+orders[2])
        eri1_d = -intor4c(mol, intor_d, aosym='s1', shls_slice=shls_slice)
        eri1_d = eri1_d.reshape(off,3,-1,naoi,naoj,naok,naol)
        eri1_d = eri1_d.transpose(1,0,2,3,4,5,6).reshape(3,-1,naoi,naoj,naok,naol)

        aoslices_d = aoslices - ao_loc[l0]
        idx_d = np.arange(naol)[None,None,None,None,None,:]
        jvp += _gen_int2e_fill_jvp_r0(eri1_d, coords_t, aoslices_d, idx_d)
    else:
        jvp += jvp_c.transpose(0,1,2,4,3)

    return jvp

@jit
def _gen_int2e_fill_jvp_r0(ints, coords_t, aoslices, aoidx):
    def _fill(sl, coord_t):
        mask = (aoidx >= sl[0]) & (aoidx < sl[1])
        grad = np.where(mask, ints, np.array(0, dtype=ints.dtype))
        return np.einsum('xyijkl,x->yijkl', grad, coord_t)
    jvp = np.sum(vmap(_fill)(aoslices, coords_t), axis=0)
    return jvp

@jit
def _int2e_dot_grad_tangent_s4(grad, tangent):
    tangent_out = np.einsum('cxijkl,x->cijkl', grad, tangent)
    tangent_out += tangent_out.transpose(0,2,1,3,4)
    tangent_out += tangent_out.transpose(0,3,4,1,2)
    return tangent_out

def _int2e_jvp_cs(mol, mol_t, intor, shls_slice, comp):
    ctr_coeff = mol.ctr_coeff
    ctr_coeff_t = mol_t.ctr_coeff

    nbas = mol.nbas
    if shls_slice is not None:
        assert shls_slice == (0, nbas, 0, nbas, 0, nbas, 0, nbas)

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
    nbas = mol.nbas
    if shls_slice is not None:
        assert shls_slice == (0, nbas, 0, nbas, 0, nbas, 0, nbas)

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

@jit
def _int2e_c2s(eris_cart, c2s):
    eris_sph = np.einsum('pi,qj,cpqrs,rk,sl->cijkl',
                         c2s, c2s, eris_cart, c2s, c2s)
    return eris_sph
