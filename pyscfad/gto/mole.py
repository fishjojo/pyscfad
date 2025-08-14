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

from functools import wraps
from pyscf.gto import mole as pyscf_mole
from pyscf.lib import logger, param
from pyscfad import numpy as np
from pyscfad import pytree
from pyscfad.gto import moleintor
from pyscfad.gto.eval_gto import eval_gto
from ._mole_helper import setup_exp, setup_ctr_coeff

Traced_Attributes = ['coords', 'exp', 'ctr_coeff', 'r0']
Exclude_Aux_Names = ('verbose',)

def inter_distance(mol=None, coords=None, Ls=None):
    """Atom distance array.

    Parameters
    ----------
    mol : :class:`Mole` instance, optional
        Either ``mol`` or ``coords`` must be specified.
    coords : array, optional
        Atom coordinates. If not specified,
        will use ``mol.atom_coords()``.
    Ls : array, optional
        Lattice translation vectors.

    Returns
    -------
    r : array
    """
    if mol is None and coords is None:
        raise KeyError("Either 'mol' or 'coords' must be specified.")
    if coords is None:
        coords = mol.atom_coords()
    rij = coords[:,None,:] - coords[None,:,:]
    if Ls is not None:
        Ls = Ls.reshape(-1, 3)
        rij = rij[None,...] + Ls[:,None,None,:]
    r2 = np.sum(rij * rij, axis=-1)
    r = np.sqrt(np.where(r2>1e-12, r2, 0))
    return r

@wraps(pyscf_mole.classical_coulomb_energy)
def classical_coulomb_energy(mol, charges=None, coords=None):
    if charges is None:
        charges = np.asarray(mol.atom_charges(), dtype=float)
    if len(charges) <= 1:
        return 0.0
    rr = inter_distance(mol, coords)
    rr = np.where(rr>1e-6, rr, np.inf)
    enuc = np.einsum('i,ij,j->', charges, 1./rr, charges) * .5
    return enuc

energy_nuc = classical_coulomb_energy

@wraps(pyscf_mole.intor_cross)
def intor_cross(intor, mol1, mol2, comp=None, grids=None):
    return moleintor.intor_cross(intor, mol1, mol2, comp=comp, grids=grids)

def nao_nr_range(mol, bas_id0, bas_id1):
    from pyscf.gto.moleintor import make_loc
    if mol.cart:
        key = 'cart'
    else:
        key = 'sph'
    ao_loc = make_loc(mol._bas[:bas_id1], key)
    nao_id0 = ao_loc[bas_id0]
    nao_id1 = ao_loc[-1]
    return nao_id0, nao_id1

class Mole(pytree.PytreeNode, pyscf_mole.Mole):
    """Subclass of :class:`pyscf.gto.Mole` with traceable attributes.

    Attributes
    ----------
    coords : array
        Atomic coordinates.
    exp : array
        Exponents of Gaussian basis functions.
    ctr_coeff : array
        Contraction coefficients of Gaussian basis functions.
    r0 : array
        Centers of Gaussian basis functions. Currently this is
        not used as the basis functions are atom centered. This
        is a placeholder for floating Gaussian basis sets.
    """
    _dynamic_attr = _keys = ['coords', 'exp', 'ctr_coeff', 'r0']

    def __init__(self, **kwargs):
        self.coords = None
        self.exp = None
        self.ctr_coeff = None
        self.r0 = None
        super().__init__(**kwargs)

    def atom_coords(self, unit='Bohr'):
        if self.coords is None:
            return np.asarray(super().atom_coords(unit))
        else:
            if not pyscf_mole.is_au(unit):
                return self.coords * param.BOHR
            else:
                return self.coords

    def set_geom_(self, atoms_or_coords, unit=None, symmetry=None,
                  inplace=True):
        mol = pyscf_mole.Mole.set_geom_(self, atoms_or_coords,
                                        unit=unit, symmetry=symmetry, inplace=inplace)
        if self.coords is not None:
            mol.coords = np.asarray(pyscf_mole.Mole.atom_coords(mol))
        if self.exp is not None:
            mol.exp = np.asarray(setup_exp(mol)[0])
        if self.ctr_coeff is not None:
            mol.ctr_coeff = np.asarray(setup_ctr_coeff(mol)[0])
        return mol

    def build(self, *args, **kwargs):
        trace_coords = kwargs.pop('trace_coords', True)
        trace_exp = kwargs.pop('trace_exp', True)
        trace_ctr_coeff = kwargs.pop('trace_ctr_coeff', True)
        trace_r0 = kwargs.pop('trace_r0', False)

        super().build(*args, **kwargs)

        if trace_coords:
            self.coords = np.asarray(self.atom_coords())
        if trace_exp:
            self.exp = np.asarray(setup_exp(self)[0])
        if trace_ctr_coeff:
            self.ctr_coeff = np.asarray(setup_ctr_coeff(self)[0])
        if trace_r0:
            raise NotImplementedError

    energy_nuc = energy_nuc
    eval_ao = eval_gto = eval_gto

    @wraps(pyscf_mole.Mole.intor)
    def intor(self, intor, comp=None, hermi=0, aosym='s1', out=None,
              shls_slice=None, grids=None):
        if not self._built:
            logger.warn(self, 'intor envs of %s not initialized.', self)
        intor = self._add_suffix(intor)
        return moleintor.intor(self, intor, comp=comp, hermi=hermi,
                               aosym=aosym, out=out, shls_slice=shls_slice,
                               grids=grids)

    def to_pyscf(self):
        mol = self.view(pyscf_mole.Mole)
        del mol.coords
        del mol.exp
        del mol.ctr_coeff
        del mol.r0
        return mol

