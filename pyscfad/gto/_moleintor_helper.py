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

from functools import lru_cache
import numpy
from pyscf.gto import mole as pyscf_mole
from pyscfad.gto import _pyscf_moleintor as moleintor

def _intor_impl(mol, intor_name, comp=None, hermi=0, aosym='s1', out=None,
                shls_slice=None, grids=None):
    bas = mol._bas
    env = mol._env
    if 'ECP' in intor_name:
        assert mol._ecp is not None
        bas = numpy.vstack((mol._bas, mol._ecpbas))
        env[pyscf_mole.AS_ECPBAS_OFFSET] = len(mol._bas)
        env[pyscf_mole.AS_NECPBAS] = len(mol._ecpbas)
        if shls_slice is None:
            shls_slice = (0, mol.nbas, 0, mol.nbas)
    elif '_grids' in intor_name:
        assert grids is not None
        env = numpy.append(env, grids.ravel())
        env[pyscf_mole.NGRIDS] = grids.shape[0]
        env[pyscf_mole.PTR_GRIDS] = env.size - grids.size

    out = moleintor.getints(intor_name, mol._atm, bas, env,
                            shls_slice, comp, hermi, aosym, out=out)
    return out

def _intor_cross_impl(intor, mol1, mol2, comp=None, grids=None):
    nbas1 = len(mol1._bas)
    nbas2 = len(mol2._bas)
    atmc, basc, envc = pyscf_mole.conc_env(mol1._atm, mol1._bas, mol1._env,
                                           mol2._atm, mol2._bas, mol2._env)
    if '_grids' in intor:
        assert grids is not None
        envc = numpy.append(envc, grids.ravel())
        envc[pyscf_mole.NGRIDS] = grids.shape[0]
        envc[pyscf_mole.PTR_GRIDS] = envc.size - grids.size

    shls_slice = (0, nbas1, nbas1, nbas1+nbas2)

    if (intor.endswith('_sph') or intor.startswith('cint') or
        intor.endswith('_spinor') or intor.endswith('_cart')):
        return moleintor.getints(intor, atmc, basc, envc, shls_slice, comp, 0)
    elif mol1.cart == mol2.cart:
        intor = mol1._add_suffix(intor)
        return moleintor.getints(intor, atmc, basc, envc, shls_slice, comp, 0)
    elif mol1.cart:
        mat = moleintor.getints(intor+'_cart', atmc, basc, envc, shls_slice, comp, 0)
        return numpy.dot(mat, mol2.cart2sph_coeff())
    else:
        mat = moleintor.getints(intor+'_cart', atmc, basc, envc, shls_slice, comp, 0)
        return numpy.dot(mol1.cart2sph_coeff().T, mat)

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

@lru_cache(100)
def index_prompt_xyz(l, increment):
    '''
    Returns the addresses of Cartesian functions with angular momentum l+increment,
    corresponding to the Cartesian functions with angular momentum l (in lexical order)
    by increasing the "increment" power for the x, y, and z components.

    For example, for Cartesian functions with l=1, the functions are x, y, and z.
    If increment=2, the computed indices correspond to the addresses of the following
    functions in the l=3 Cartesian functions:
    [xxx, xxy, xxz]
    [yyx, yyy, yyz]
    [zzx, zzy, zzz]
    '''
    l2 = l + increment
    l2_addrs = {}
    n = 0
    for x in range(l2, -1, -1):
        for y in range(l2-x, -1, -1):
            z = l2-x-y
            l2_addrs[x,y,z] = n
            n += 1

    x_idx = []
    y_idx = []
    z_idx = []
    for x in range(l, -1, -1):
        for y in range(l-x, -1, -1):
            z = l-x-y
            x_idx.append(l2_addrs[x+increment,y,z])
            y_idx.append(l2_addrs[x,y+increment,z])
            z_idx.append(l2_addrs[x,y,z+increment])
    return x_idx, y_idx, z_idx

def int1e_get_dr_order(intor):
    fname = intor.replace('_sph', '').replace('_cart', '')
    if fname[-4:-2] == 'dr':
        orders = [int(fname[-2]), int(fname[-1])]
    else:
        orders = [0, 0]
    return orders

def int2e_get_dr_order(intor):
    fname = intor.replace('_sph', '').replace('_cart', '')
    if fname[-6:-4] == 'dr':
        orders = [int(fname[-4]), int(fname[-3]), int(fname[-2]), int(fname[-1])]
    else:
        orders = [0,] * 4
    return orders

def int1e_dr1_name(intor):
    if 'sph' in intor:
        suffix = '_sph'
    elif 'cart' in intor:
        suffix = '_cart'
    else:
        suffix = ''
    fname = intor.replace('_sph', '').replace('_cart', '')

    # special cases first
    #if fname == 'int1e_r':
    #    intor_ip_bra = None
    #    intor_ip_ket = 'int1e_irp' + suffix
    if fname[-4:-2] == 'dr':
        orders = [int(fname[-2]), int(fname[-1])]
        intor_ip_bra = fname[:-2] + str(orders[0]+1) + str(orders[1]) + suffix
        intor_ip_ket = fname[:-2] + str(orders[0]) + str(orders[1]+1) + suffix
    else:
        intor_ip_bra = fname + '_dr10' + suffix
        intor_ip_ket = fname + '_dr01' + suffix
    return intor_ip_bra, intor_ip_ket

def int2e_dr1_name(intor):
    if 'sph' in intor:
        suffix = '_sph'
    elif 'cart' in intor:
        suffix = '_cart'
    else:
        suffix = ''
    fname = intor.replace('_sph', '').replace('_cart', '')

    if fname[-6:-4] == 'dr':
        orders = int2e_get_dr_order(intor)
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
