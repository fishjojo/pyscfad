from pyscf import gto
from pyscfad import lib
from pyscfad.lib import np_helper as np
from . import moleintor

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
