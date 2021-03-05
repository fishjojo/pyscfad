from jax import numpy as jnp
from flax import struct
from pyscf.gto import mole as pmole
from . import moleintor

@struct.dataclass
class Mole(pmole.Mole):
    mol: pmole.Mole = struct.field(pytree_node=False)
    coords: jnp.array = None
    exponents: jnp.array = None
    contract_coeff: jnp.array = None

    def __post_init__(self):
        for key, value in self.mol.__dict__.items():
            object.__setattr__(self, key, value) # copy attributes of mol

    def setattr(self, attr, value):
        """
        Mimics :func:`setattr`

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
