from jax import numpy as jnp
from flax import struct
from pyscf.gto import mole
from . import moleintor

@struct.dataclass
class Mole():
    mol: mole.Mole = struct.field(pytree_node=False)
    coords: jnp.array = None
    exponents: jnp.array = None
    contract_coeff: jnp.array = None

    def intor(self, intor, comp=None, hermi=0, aosym='s1', out=None,
              shls_slice=None):
        if intor=="int1e_ovlp":
            return moleintor.int1e_ovlp(self)
        else:
            raise NotImplementedError

