from functools import partial
import numpy
from jax import custom_jvp
from pyscf.gto import mole, moleintor
from pyscf.gto.mole import Mole
from pyscfad.lib import numpy as np
from pyscfad.lib import ops
from ._mole_helper import uncontract, setup_exp, setup_ctr_coeff

def getints(mol, intor, shls_slice=None,
            comp=None, hermi=0, aosym='s1', out=None):
    if intor.endswith("_spinor"):
        raise NotImplementedError

    if (intor.startswith("int1e") or
        intor.startswith('ECP')):
        return getints2c(mol, intor, shls_slice, comp, hermi, aosym, out)
    elif intor.startswith("int2e"):
        return getints4c(mol, intor, shls_slice, comp, aosym, out)
    else:
        raise NotImplementedError

@partial(custom_jvp, nondiff_argnums=tuple(range(1,7)))
def getints2c(mol, intor,
              shls_slice=None, comp=1, hermi=0, aosym='s1', out=None):
    return Mole.intor(mol, intor,
                      comp=comp, hermi=hermi, aosym=aosym,
                      shls_slice=shls_slice, out=out)

@getints2c.defjvp
def getints2c_jvp(intor, shls_slice, comp, hermi, aosym, out,
                  primals, tangents):
    if shls_slice is not None:
        raise NotImplementedError

    mol, = primals
    primal_out = getints2c(mol, intor, shls_slice=None,
                           comp=comp, hermi=hermi, aosym=aosym, out=out)

    mol_t, = tangents
    tangent_out = np.zeros_like(primal_out)
    if mol.coords is not None:
        if intor.startswith("ECPscalar"):
            intor_ip = intor.replace("ECPscalar", "ECPscalar_ipnuc")
        else:
            tmp = intor.split("_", 1)
            intor_ip = tmp[0] + "_ip" + tmp[1]
        tangent_out += _int1e_jvp_r0(mol, mol_t, intor_ip)

        if intor.startswith("int1e_nuc"):
            intor_ip = intor.replace("int1e_nuc", "int1e_iprinv")
        elif intor.startswith("ECPscalar"):
            intor_ip = intor.replace("ECPscalar", "ECPscalar_iprinv")
        tangent_out += _int1e_nuc_jvp_rc(mol, mol_t, intor_ip)
    if mol.ctr_coeff is not None:
        tangent_out += _int1e_jvp_cs(mol, mol_t, intor)
    if mol.exp is not None:
        tangent_out += _int1e_jvp_exp(mol, mol_t, intor)
    return primal_out, tangent_out

@partial(custom_jvp, nondiff_argnums=tuple(range(1,6)))
def getints4c(mol, intor,
              shls_slice=None, comp=1, aosym='s1', out=None):
    return Mole.intor(mol, intor,
                      comp=comp, aosym=aosym,
                      shls_slice=shls_slice, out=out)

@getints4c.defjvp
def getints4c_jvp(intor, shls_slice, comp, aosym, out,
                  primals, tangents):
    if shls_slice is not None:
        raise NotImplementedError
    mol, = primals
    primal_out = getints4c(mol, intor, shls_slice, comp, aosym, out)

    mol_t, = tangents
    tangent_out = np.zeros_like(primal_out)

    if mol.coords is not None:
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
    coords_t = mol_t.coords
    atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()
    nao = mol.nao
    tangent_out = np.zeros((nao,nao))
    s1 = -Mole.intor(mol, intor, comp=3)
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices [ia,2:]
        tmp = np.einsum('xij,x->ij',s1[:,p0:p1],coords_t[k])
        tangent_out = ops.index_add(tangent_out, ops.index[p0:p1], tmp)
        tangent_out = ops.index_add(tangent_out, ops.index[:,p0:p1], tmp.T)
    return tangent_out

def _int1e_nuc_jvp_rc(mol, mol_t, intor):
    coords_t = mol_t.coords
    atmlst = range(mol.natm)
    nao = mol.nao
    tangent_out = np.zeros((nao,nao))
    for k, ia in enumerate(atmlst):
        with mol.with_rinv_at_nucleus(ia):
            vrinv = Mole.intor(mol, intor, comp=3)
            if "ECP" not in intor:
                vrinv *= -mol.atom_charge(ia)
        vrinv += vrinv.transpose(0,2,1)
        tangent_out += np.einsum('xij,x->ij', vrinv, coords_t[k])
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
    grad = np.zeros((len(ctr_coeff), nao, nao), dtype=float)

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
            idx = ops.index[(cs_of[i]+j*nprim):(cs_of[i]+(j+1)*nprim), ibas:(ibas+nbas), :]
            grad = ops.index_add(grad, idx, g)
            ibas += nbas
        off += nprim*nbas
    grad += grad.transpose(0,2,1)

    tangent_out = np.einsum('xij,x->ij', grad, ctr_coeff_t)
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
    grad = np.zeros((len(es), nao, nao), dtype=float)

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
                    grad = ops.index_add(grad, ops.index[ioff+j, jbas], gc)
                    jbas += 1
            ibas += nbas
        off += nprim * nbas1

    tangent_out = np.einsum('xij,x->ij', grad, mol_t.exp)
    tangent_out += tangent_out.T
    if not mol.cart or not cart:
        c2s = np.asarray(mol.cart2sph_coeff())
        tangent_out = np.dot(c2s.T, np.dot(tangent_out, c2s))
    return tangent_out

def _int2e_jvp_r0(mol, mol_t, intor):
    coords = mol_t.coords
    atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()
    nao = mol.nao
    tangent_out = np.zeros([nao,]*4)

    eri1 = -Mole.intor(mol, intor, comp=3)
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices [ia,2:]
        tmp = np.einsum("xijkl,x->ijkl", eri1[:,p0:p1], coords[k])
        tangent_out = ops.index_add(tangent_out, ops.index[p0:p1], tmp)
        tangent_out = ops.index_add(tangent_out, ops.index[:,p0:p1], tmp.transpose((1,0,2,3)))
        tangent_out = ops.index_add(tangent_out, ops.index[:,:,p0:p1], tmp.transpose((2,3,0,1)))
        tangent_out = ops.index_add(tangent_out, ops.index[:,:,:,p0:p1], tmp.transpose((2,3,1,0)))
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
    grad = np.zeros((len(ctr_coeff), nao, nao, nao, nao), dtype=float)

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
            idx = ops.index[(cs_of[i]+j*nprim):(cs_of[i]+(j+1)*nprim), ibas:(ibas+nbas)]
            grad = ops.index_add(grad, idx, g)
            ibas += nbas
        off += nprim*nbas
    tangent_out = np.einsum('xijkl,x->ijkl', grad, ctr_coeff_t)
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
    grad = np.zeros((len(es), nao, nao, nao, nao), dtype=float)

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
                    grad = ops.index_add(grad, ops.index[ioff+j, jbas], gc)
                    jbas += 1
            ibas += nbas
        off += nprim * nbas1

    tangent_out = np.einsum('xijkl,x->ijkl', grad, mol_t.exp)
    tangent_out += tangent_out.transpose(1,0,2,3)
    tangent_out += tangent_out.transpose(2,3,0,1)
    if not mol.cart:
        c2s = np.asarray(mol.cart2sph_coeff())
        tangent_out = np.einsum("iu,jv,ijkl,ks,lt->uvst", c2s, c2s, tangent_out, c2s, c2s)
    return tangent_out
