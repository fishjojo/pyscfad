from functools import wraps
from pyscf.gto import mole as pyscf_mole
from pyscf.lib import logger, param
from pyscfad import numpy as np
from pyscfad import util
from pyscfad.gto import moleintor
from pyscfad.gto.eval_gto import eval_gto
from ._mole_helper import setup_exp, setup_ctr_coeff

Traced_Attributes = ['coords', 'exp', 'ctr_coeff', 'r0']

@wraps(pyscf_mole.inter_distance)
def inter_distance(mol, coords=None):
    if coords is None:
        coords = mol.atom_coords()
    r = coords[:,None,:] - coords[None,:,:]
    rr = np.sum(r*r, axis=2)
    rr = np.sqrt(np.where(rr>1e-10, rr, 0))
    return rr

@wraps(pyscf_mole.classical_coulomb_energy)
def classical_coulomb_energy(mol, charges=None, coords=None):
    if charges is None:
        charges = np.asarray(mol.atom_charges(), dtype=float)
    if len(charges) <= 1:
        return 0.0
    rr = inter_distance(mol, coords)
    rr = np.where(rr>1e-5, rr, 1e200)
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

@util.pytree_node(Traced_Attributes)
class Mole(pyscf_mole.Mole):
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

    _keys = {'coords', 'exp', 'ctr_coeff', 'r0'}

    def __init__(self, **kwargs):
        self.coords = None
        self.exp = None
        self.ctr_coeff = None
        self.r0 = None
        super().__init__(**kwargs)

    def atom_coords(self, unit='Bohr'):
        if self.coords is None:
            return super().atom_coords(unit)
        else:
            if unit[:3].upper() == 'ANG':
                return self.coords * param.BOHR
            else:
                return self.coords

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
