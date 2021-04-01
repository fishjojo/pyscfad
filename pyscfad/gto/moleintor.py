from jax import custom_jvp
from pyscf.gto import mole, moleintor
from pyscf.gto.mole import Mole
from pyscfad.lib import numpy as np
from pyscfad.lib import ops
from pyscfad import gto

def getints(mol, intor):
    if intor == "int1e_ovlp":
        return int1e_ovlp(mol)
    elif intor == "int1e_kin":
        return int1e_kin(mol)
    elif intor == "int1e_nuc":
        return int1e_nuc(mol)
    elif intor == "ECPscalar":
        return ECPscalar(mol)
    elif intor == "int2e":
        return int2e(mol)
    else:
        raise NotImplementedError

def int1e_2c_nuc_jvp(mol, mol_t, intor):
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

def get_fakemol_cs(mol):
    mol1 = gto.mole.uncontract(mol)
    return mol1

def get_fakemol_exp(mol):
    mol1 = gto.mole.uncontract(mol)
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


def int1e_2c_cs_jvp(mol, mol_t, intor, factor=1.):
    ctr_coeff = mol.ctr_coeff
    ctr_coeff_t = mol_t.ctr_coeff

    mol1 = get_fakemol_cs(mol)
    s = mole.intor_cross(intor, mol1, mol) * factor
    _, cs_of, _ = gto.mole.setup_ctr_coeff(mol)
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

def int1e_2c_exp_jvp(mol, mol_t, intor, factor=1.):
    mol1 = get_fakemol_exp(mol)
    intor = mol._add_suffix(intor, cart=True)
    s = mole.intor_cross(intor, mol1, mol) * factor
    es, es_of, _env_of = gto.mole.setup_exp(mol)
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

    grad += grad.transpose(0,2,1)
    if not mol.cart:
        c2s = np.asarray(mol.cart2sph_coeff())
        grad = np.einsum("iu,xij,jv->xuv", c2s, grad, c2s)
    tangent_out = np.einsum('xij,x->ij', grad, mol_t.exp)
    return tangent_out

def int1e_2c_r0_jvp():
    pass

@custom_jvp
def int1e_ovlp(mol):
    return Mole.intor(mol, "int1e_ovlp")

@int1e_ovlp.defjvp
def int1e_ovlp_jvp(primals, tangents):
    mol, = primals
    primal_out = int1e_ovlp(mol)

    mol_t, = tangents
    tangent_out = np.zeros_like(primal_out)
    if mol.coords is not None:
        tangent_out += int1e_2c_nuc_jvp(mol, mol_t, "int1e_ipovlp")
    if mol.ctr_coeff is not None:
        tangent_out += int1e_2c_cs_jvp(mol, mol_t, "int1e_ovlp")
    if mol.exp is not None:
        tangent_out += int1e_2c_exp_jvp(mol, mol_t, "int1e_ovlp")
    return primal_out, tangent_out

@custom_jvp
def int1e_kin(mol):
    return Mole.intor(mol, "int1e_kin")

@int1e_kin.defjvp
def int1e_kin_jvp(primals, tangents):
    mol, = primals
    primal_out = int1e_kin(mol)

    mol_t, = tangents
    tangent_out = np.zeros_like(primal_out)
    if mol.coords is not None:
        tangent_out += int1e_2c_nuc_jvp(mol, mol_t, "int1e_ipkin")
    if mol.ctr_coeff is not None:
        tangent_out += int1e_2c_cs_jvp(mol, mol_t, "int1e_kin")
    if mol.exp is not None:
        tangent_out += int1e_2c_exp_jvp(mol, mol_t, "int1e_kin")
    return primal_out, tangent_out

def int1e_nuc_nuc_jvp(mol, mol_t):
    coords = mol_t.coords
    atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()
    nao = mol.nao
    tangent_out = np.zeros((nao,nao))
    h1 = -Mole.intor(mol, 'int1e_ipnuc', comp=3)
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices [ia,2:]
        with mol.with_rinv_at_nucleus(ia):
            vrinv = Mole.intor(mol, 'int1e_iprinv', comp=3)
            vrinv *= -mol.atom_charge(ia)
        vrinv[:,p0:p1] += h1[:,p0:p1]
        tmp = vrinv + vrinv.transpose(0,2,1)
        tmp1 = np.einsum('xij,x->ij',tmp, coords[k])
        tangent_out = ops.index_add(tangent_out, ops.index[:,:], tmp1)
    return tangent_out

@custom_jvp
def int1e_nuc(mol):
    return Mole.intor(mol, "int1e_nuc")

@int1e_nuc.defjvp
def int1e_nuc_jvp(primals, tangents):
    mol, = primals
    primal_out = int1e_nuc(mol)

    mol_t, = tangents
    tangent_out = np.zeros_like(primal_out)

    if mol.coords is not None:
        tangent_out += int1e_nuc_nuc_jvp(mol, mol_t)
    if mol.ctr_coeff is not None:
        tangent_out += int1e_2c_cs_jvp(mol, mol_t, "int1e_nuc", factor=0.5)
    if mol.exp is not None:
        tangent_out += int1e_2c_exp_jvp(mol, mol_t, "int1e_nuc", factor=0.5)
    return primal_out, tangent_out

@custom_jvp
def ECPscalar(mol):
    return Mole.intor(mol, "ECPscalar")

@ECPscalar.defjvp
def ECPscalar_jvp(primals, tangents):
    mol, = primals
    primal_out = ECPscalar(mol)

    mol_t, = tangents
    coords = mol_t.coords
    atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()
    nao = mol.nao
    tangent_out = np.zeros((nao,nao))

    h1 = -Mole.intor(mol, 'ECPscalar_ipnuc', comp=3)
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices [ia,2:]
        with mol.with_rinv_at_nucleus(ia):
            vrinv = Mole.intor(mol, 'ECPscalar_iprinv', comp=3)
        vrinv[:,p0:p1] += h1[:,p0:p1]
        tmp = vrinv + vrinv.transpose(0,2,1)
        tmp1 = np.einsum('xij,x->ij',tmp, coords[k])
        tangent_out = ops.index_add(tangent_out, ops.index[:,:], tmp1)
    return primal_out, tangent_out

def int2e_nuc_jvp(mol, mol_t):
    coords = mol_t.coords
    atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()
    nao = mol.nao
    tangent_out = np.zeros([nao,]*4)

    eri1 = -Mole.intor(mol, "int2e_ip1", comp=3)
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices [ia,2:]
        tmp = np.einsum("xijkl,x->ijkl", eri1[:,p0:p1], coords[k])
        tangent_out = ops.index_add(tangent_out, ops.index[p0:p1], tmp)
        tangent_out = ops.index_add(tangent_out, ops.index[:,p0:p1], tmp.transpose((1,0,2,3)))
        tangent_out = ops.index_add(tangent_out, ops.index[:,:,p0:p1], tmp.transpose((2,3,0,1)))
        tangent_out = ops.index_add(tangent_out, ops.index[:,:,:,p0:p1], tmp.transpose((2,3,1,0)))
    return tangent_out

def int2e_cs_jvp(mol, mol_t, intor):
    ctr_coeff = mol.ctr_coeff
    ctr_coeff_t = mol_t.ctr_coeff

    mol1 = get_fakemol_cs(mol)

    nbas = len(mol._bas)
    nbas1 = len(mol1._bas)
    atmc, basc, envc = mole.conc_env(mol1._atm, mol1._bas, mol1._env,
                                     mol._atm, mol._bas, mol._env)
    shls_slice = (0, nbas1, nbas1, nbas1+nbas, nbas1, nbas1+nbas, nbas1, nbas1+nbas)
    intor = mol._add_suffix(intor)
    eri = moleintor.getints(intor, atmc, basc, envc, shls_slice)

    _, cs_of, _ = gto.mole.setup_ctr_coeff(mol)
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

def int2e_exp_jvp(mol, mol_t, intor="int2e"):
    mol1 = get_fakemol_exp(mol)
    intor = mol._add_suffix(intor, cart=True)

    nbas = len(mol._bas)
    nbas1 = len(mol1._bas)
    atmc, basc, envc = mole.conc_env(mol1._atm, mol1._bas, mol1._env,
                                     mol._atm, mol._bas, mol._env)
    shls_slice = (0, nbas1, nbas1, nbas1+nbas, nbas1, nbas1+nbas, nbas1, nbas1+nbas)

    eri = moleintor.getints(intor, atmc, basc, envc, shls_slice)

    es, es_of, _env_of = gto.mole.setup_exp(mol)
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

@custom_jvp
def int2e(mol):
    return Mole.intor(mol, "int2e")

@int2e.defjvp
def int2e_jvp(primals, tangents):
    mol, = primals
    primal_out = int2e(mol)

    mol_t, = tangents
    tangent_out = np.zeros_like(primal_out)

    if mol.coords is not None:
        tangent_out += int2e_nuc_jvp(mol, mol_t)
    if mol.ctr_coeff is not None:
        tangent_out += int2e_cs_jvp(mol, mol_t, "int2e")
    if mol.exp is not None:
        tangent_out += int2e_exp_jvp(mol, mol_t, "int2e")
    return primal_out, tangent_out
