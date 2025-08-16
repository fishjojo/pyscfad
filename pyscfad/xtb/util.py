# Copyright 2021-2025 Xing Zhang
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

import numpy
from pyscf.gto.mole import ANG_OF
from pyscf.data.elements import NUC
from pyscfad import numpy as np
from pyscfad import ops
from pyscfad.lib import unpack_tril

ANG_MOMENT = {
  -20: "2S", # H 2s
    0: "s",
    1: "p",
    2: "d",
    3: "f",
    4: "g",
}

SHELL_ORDER = {
    "s": 0,
    "p": 1,
    "d": 2,
    "f": 3,
    "g": 4,
}

def aoslice_by_shell(mol):
    ao_loc = mol.ao_loc
    return numpy.stack([ao_loc[:-1], ao_loc[1:]], axis=1)

def unique_element_to_atom_indices(mol):
    uniq_elements, indices = numpy.unique(mol.elements, return_inverse=True)
    return uniq_elements, indices

def atom_to_bas_indices(mol):
    natm = mol.natm
    nbas_per_atom = [mol.atom_nshells(ia) for ia in range(natm)]
    indices = numpy.repeat(numpy.arange(natm), nbas_per_atom)
    return indices

def atom_to_bas_indices_2d(mol):
    atm_to_bas_id = atom_to_bas_indices(mol)
    return numpy.ix_(atm_to_bas_id, atm_to_bas_id)

def bas_to_ao_indices(mol):
    ao_loc = mol.ao_loc
    nao_per_shell = ao_loc[1:] - ao_loc[:-1]
    indices = numpy.repeat(numpy.arange(mol.nbas), nao_per_shell)
    return indices

def bas_to_ao_indices_2d(mol):
    bas_to_ao_id = bas_to_ao_indices(mol)
    return numpy.ix_(bas_to_ao_id, bas_to_ao_id)

def atom_to_ao_indices(mol):
    atm_to_bas_id = atom_to_bas_indices(mol)
    bas_to_ao_id = bas_to_ao_indices(mol)
    return atm_to_bas_id[bas_to_ao_id]

def atom_to_ao_indices_2d(mol):
    atm_to_ao_id = atom_to_ao_indices(mol)
    return numpy.ix_(atm_to_ao_id, atm_to_ao_id)

def unique_l_to_bas_indices(mol):
    uniq_l, indices = numpy.unique(mol._bas[:,ANG_OF], return_inverse=True)
    return uniq_l, indices

def unique_bas_to_bas_indices(mol):
    elements, uniq_elem_to_atm_id = unique_element_to_atom_indices(mol)

    uniq_element_nbas = {}
    #nbas_per_elem = []
    indices = numpy.empty(mol.nbas, dtype=numpy.int32)
    i1 = 0
    for ia, elem in enumerate(elements):
        atm_ids = numpy.where(ia == uniq_elem_to_atm_id)[0]
        nbas = mol.atom_nshells(atm_ids[0])
        #nbas_per_elem.append(nbas)
        uniq_element_nbas[elem] = nbas
        i0, i1 = i1, i1 + nbas
        for i in atm_ids:
            bas_ids = mol.atom_shell_ids(i)
            indices[bas_ids] = numpy.arange(i0, i1)
    #return elements, nbas_per_elem, indices
    return uniq_element_nbas, indices

def load_global_params(param, name):
    attrs = name.split(".")
    for attr in attrs:
        param = param.get(attr)
    return param

def load_unique_element_params(
    mol, param, name,
    broadcast=None,
):
    param_element = getattr(param, "element")

    uniq_elements, uniq_elem_to_atm_id = unique_element_to_atom_indices(mol)
    vals = [getattr(param_element[symb], name) for symb in uniq_elements]
    vals = np.asarray(vals)

    if broadcast is None:
        return vals, uniq_elem_to_atm_id
    elif isinstance(broadcast, str):
        broadcast = broadcast.lower()
        if broadcast == "atom":
            return vals[uniq_elem_to_atm_id]
        elif broadcast == "shell":
            atm_to_bas_id = atom_to_bas_indices(mol)
            return vals[uniq_elem_to_atm_id[atm_to_bas_id]]
        elif broadcast == "ao":
            atm_to_ao_id = atom_to_ao_indices(mol)
            return vals[uniq_elem_to_atm_id[atm_to_ao_id]]
        else:
            raise KeyError(f"Unsupported broadcasting type {broadcast}")

def _sort_key_for_shell(s):
    n = int(s[:-1])
    l = s[-1]
    return (SHELL_ORDER[l], n)

def load_unique_element_shell_params(
    mol, param, name,
    pad=0,
    broadcast=None,
):
    param_element = getattr(param, "element")
    #elements, nbas_per_elem, uniq_bas_to_bas_id = unique_bas_to_bas_indices(mol)
    uniq_element_nbas, uniq_bas_to_bas_id = unique_bas_to_bas_indices(mol)

    vals = []
    #for elem, nbas in zip(elements, nbas_per_elem):
    for elem, nbas in uniq_element_nbas.items():
        shells = getattr(param_element[elem], "shells")

        #sorted_shell_id = numpy.asarray(sorted(range(len(shells)),
        #                                       key=lambda i: _sort_key_for_shell(shells[i])))

        keys = numpy.asarray([_sort_key_for_shell(s) for s in shells])
        sorted_shell_id = numpy.lexsort((keys[:,1],keys[:,0]))

        val = np.asarray(getattr(param_element[elem], name))[sorted_shell_id]
        assert nbas == len(val), "Inconsistent basis set and parameter set"
        #npad = nbas - len(val)
        #if npad > 0:
        #    # FIXME should not pad
        #    val = np.pad(val, (0, npad), constant_values=pad)
        #else:
        #    val = val[:nbas]
        vals.append(val)
    vals = np.concatenate(vals, axis=None)

    if broadcast is None:
        return vals, uniq_bas_to_bas_id
    elif isinstance(broadcast, str):
        broadcast = broadcast.lower()
        if broadcast == "shell":
            return vals[uniq_bas_to_bas_id]
        elif broadcast == "ao":
            bas_to_ao_id = bas_to_ao_indices(mol)
            uniq_bas_to_ao_id = uniq_bas_to_bas_id[bas_to_ao_id]
            return vals[uniq_bas_to_ao_id]
        else:
            raise KeyError(f"Unsupported broadcasting type {broadcast}")

def load_global_element_pair_params(
    mol, param, name,
    pad=1.,
    broadcast=None,
):
    param_kpair = getattr(param, name)
    elements, uniq_elem_to_atm_id = unique_element_to_atom_indices(mol)
    keys = []
    for ia, symb_A in enumerate(elements):
        for symb_B in elements[:ia+1]:
            if NUC[symb_A] >= NUC[symb_B]:
                key = f"{symb_A}-{symb_B}"
            else:
                key = f"{symb_B}-{symb_A}"
            keys.append(key)

    val = [param_kpair.get(key, pad) for key in keys]
    val = np.asarray(val, dtype=float)

    if broadcast is None:
        return keys, val
    elif isinstance(broadcast, str):
        val_mat = unpack_tril(val)
        broadcast = broadcast.lower()
        if broadcast == "atom":
            uniq_elem_to_atm_id_2d = np.ix_(uniq_elem_to_atm_id, uniq_elem_to_atm_id)
            return val_mat[uniq_elem_to_atm_id_2d]
        elif broadcast == "shell":
            atm_to_bas_id = atom_to_bas_indices(mol)
            uniq_elem_to_bas_id = uniq_elem_to_atm_id[atm_to_bas_id]
            uniq_elem_to_bas_id_2d = np.ix_(uniq_elem_to_bas_id, uniq_elem_to_bas_id)
            return val_mat[uniq_elem_to_bas_id_2d]
        elif broadcast == "ao":
            atm_to_ao_id = atom_to_ao_indices(mol)
            uniq_elem_to_ao_id = uniq_elem_to_atm_id[atm_to_ao_id]
            uniq_elem_to_ao_id_2d = np.ix_(uniq_elem_to_ao_id, uniq_elem_to_ao_id)
            return val_mat[uniq_elem_to_ao_id_2d]
        else:
            raise KeyError(f"Unsupported broadcasting type {broadcast}")

def load_global_shell_pair_params(mol, param, name, pad=1., broadcast=None,
                                  ls=None, uniq_l_to_bas_id=None):
    param_shell = getattr(param, name)
    if ls is None or uniq_l_to_bas_id is None:
        ls, uniq_l_to_bas_id = unique_l_to_bas_indices(mol)
    keys = []
    for i, li in enumerate(ls):
        for lj in ls[:i+1]:
            key = ANG_MOMENT[min(li, lj)] + ANG_MOMENT[max(li, lj)]
            keys.append(key)
    val = [param_shell.get(key, pad) for key in keys]
    val = np.asarray(val, dtype=float)

    if broadcast is None:
        return keys, val
    elif isinstance(broadcast, str):
        val_mat = unpack_tril(val)
        broadcast = broadcast.lower()
        if broadcast == "shell":
            uniq_l_to_bas_id_2d = np.ix_(uniq_l_to_bas_id, uniq_l_to_bas_id)
            return val_mat[uniq_l_to_bas_id_2d]
        elif broadcast == "ao":
            bas_to_ao_id = bas_to_ao_indices(mol)
            uniq_l_to_ao_id = uniq_l_to_bas_id[bas_to_ao_id]
            uniq_l_to_ao_id_2d = np.ix_(uniq_l_to_ao_id, uniq_l_to_ao_id)
            return val_mat[uniq_l_to_ao_id_2d]
        else:
            raise KeyError(f"Unsupported broadcasting type {broadcast}")

def load_global_shell_pair_params_gfn1(mol, param, name, pad=1., broadcast=None):
    # GFN1 treat H 2s shell differently
    mask = mask_valence_shell_gfn1(mol)
    fakemol = mol.copy()
    fakemol._bas[:,ANG_OF][~mask] = -20
    ls, uniq_l_to_bas_id = unique_l_to_bas_indices(fakemol)
    return load_global_shell_pair_params(mol, param, name, pad=pad, broadcast=broadcast,
                                         ls=ls, uniq_l_to_bas_id=uniq_l_to_bas_id)

def mask_valence_shell_gfn1(mol):
    mask = numpy.ones(mol.nbas, dtype=bool)
    if not "H" in mol.elements:
        return mask

    # deal with H 2s shell
    elements, uniq_elem_to_atm_id = unique_element_to_atom_indices(mol)
    H_id = numpy.where(elements=="H")[0][0]

    atm_to_bas_id = atom_to_bas_indices(mol)
    uniq_elem_to_bas_id = uniq_elem_to_atm_id[atm_to_bas_id]
    H_shell_id = numpy.where(uniq_elem_to_bas_id==H_id)[0]

    mask[H_shell_id[1::2]] = False
    return mask

def mask_atom_pairs(mol, exclude_diag=True):
    mask = numpy.ones((mol.natm, mol.natm), dtype=bool)
    if exclude_diag:
        numpy.fill_diagonal(mask, False)
    return mask

def rcut_erfc(alpha, q, precision=1e-8):
    r"""Cutoff radius for :math:`E = q^2 erfc(\alpha r)/r`.

    .. math::

        4\pi q^2 \int_{r_0}^{\infty} r erfc(\alpha r) dr < \epsilon
    """
    alpha = ops.to_numpy(alpha)
    q = ops.to_numpy(q)
    q = numpy.maximum(q, 0.1) #small charge cause divergence
    q2 = q * q
    a2 = alpha * alpha
    a3 = a2 * alpha
    fac = precision / numpy.sqrt(numpy.pi) / q2

    r0 = 20.
    r0 = numpy.sqrt(-numpy.log(a3 * r0 * fac) / a2)
    r0 = numpy.sqrt(-numpy.log(a3 * r0 * fac) / a2)
    return r0

