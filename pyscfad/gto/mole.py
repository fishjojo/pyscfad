import jax
from pyscf import gto
from pyscfad import lib
from pyscfad.lib import np_helper as np
from . import moleintor


def energy_nuc(mol, charges=None, coords=None):
    if charges is None: charges = mol.atom_charges()
    if len(charges) <= 1:
        return 0
    rr = inter_distance(mol, coords)
    rr = jax.ops.index_update(rr, np.diag_indices_from(rr), 1.e200)
    e = np.einsum('i,ij,j->', charges, 1./rr, charges) * .5
    return e

def inter_distance(mol, coords=None):
    if coords is None: coords = mol.coords
    rr = np.linalg.norm(coords.reshape(-1,1,3) - coords, axis=2)
    rr = jax.ops.index_update(rr, np.diag_indices_from(rr), 0.)
    return rr


@lib.dataclass
class Mole(gto.Mole):
    mol: gto.Mole
    coords: np.array = lib.field(pytree_node=True, default=None)
    exponents: np.array = lib.field(pytree_node=True, default=None)
    contract_coeff: np.array = lib.field(pytree_node=True, default=None)

    def __post_init__(self):
        # copy the attributes of self.mol
        for key, value in self.mol.__dict__.items():
            object.__setattr__(self, key, value)

    def __setattr__(self, attr, value):
        """
        Note:
            This function modifies the attributes of both `self` and `self.mol`
        """
        object.__setattr__(self, attr, value)
        if attr in self.mol.__dict__:
            setattr(self.mol, attr, value)

    energy_nuc = energy_nuc

    def intor(self, intor, comp=None, hermi=0, aosym='s1', out=None,
              shls_slice=None):
        if intor == "int1e_ovlp":
            return moleintor.int1e_ovlp(self)
        elif intor == "int1e_kin":
            return moleintor.int1e_kin(self)
        elif intor == "int1e_nuc":
            return moleintor.int1e_nuc(self)
        elif intor == "ECPscalar":
            return moleintor.ECPscalar(self)
        elif intor == "int2e":
            return moleintor.int2e(self)
        else:
            raise NotImplementedError
