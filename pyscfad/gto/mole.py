import sys
from typing import Optional, Union, Any
import numpy

from pyscf import __config__
from pyscf import gto
from pyscf.lib import logger, param
from pyscf.gto.mole import PTR_ENV_START

from pyscfad import lib
from pyscfad.lib import numpy as jnp
from pyscfad.lib import ops
from pyscfad.gto import moleintor

def energy_nuc(mol, charges=None, coords=None):
    if charges is None: charges = mol.atom_charges()
    if len(charges) <= 1:
        return 0
    rr = inter_distance(mol, coords)
    rr = ops.index_update(rr, jnp.diag_indices_from(rr), 1.e200)
    e = jnp.einsum('i,ij,j->', charges, 1./rr, charges) * .5
    return e

def inter_distance(mol, coords=None):
    if coords is None: coords = mol.coords
    if coords is None: coords = mol.atom_coords()
    rr = jnp.linalg.norm(coords.reshape(-1,1,3) - coords, axis=2)
    rr = ops.index_update(rr, jnp.diag_indices_from(rr), 0.)
    return rr


@lib.dataclass
class Mole(gto.Mole):
    # traced attributes
    # NOTE jax requires that at least one variable needs to be traced for AD
    coords: jnp.array = lib.field(pytree_node=True, default=jnp.zeros([1,3], dtype=float))
    exponents: Optional[jnp.array] = lib.field(pytree_node=True, default=None)
    contract_coeff: Optional[jnp.array] = lib.field(pytree_node=True, default=None)

    # attributes of the base class
    verbose: int = lib.field(default = getattr(__config__, 'VERBOSE', logger.NOTE))
    unit: str = lib.field(default = getattr(__config__, 'UNIT', 'angstrom'))
    incore_anyway: bool = lib.field(default = getattr(__config__, 'INCORE_ANYWAY', False))
    cart: bool = lib.field(default = getattr(__config__, 'gto_mole_Mole_cart', False))

    # attributes of the base class object
    output: Optional[str] = lib.field(default = None)
    max_memory: int = lib.field(default = param.MAX_MEMORY)
    charge: int = lib.field(default = 0)
    spin: int = lib.field(default = 0)
    symmetry: bool = lib.field(default = False)
    symmetry_subgroup: Optional[str] = lib.field(default = None)
    cart: bool = lib.field(default = False)
    atom: Union[list,str] = lib.field(default_factory = list)
    basis: Union[dict,str] = lib.field(default = 'sto-3g')
    nucmod: Union[dict,str] = lib.field(default_factory = dict)
    ecp: Union[dict,str] = lib.field(default_factory = dict)
    nucprop: dict = lib.field(default_factory = dict)

    # private attributes
    _atm: numpy.ndarray = lib.field(default = numpy.zeros((0,6), dtype=numpy.int32))
    _bas: numpy.ndarray = lib.field(default = numpy.zeros((0,8), dtype=numpy.int32))
    _env: numpy.ndarray = lib.field(default = numpy.zeros(PTR_ENV_START))
    _ecpbas: numpy.ndarray = lib.field(default = numpy.zeros((0,8), dtype=numpy.int32))

    stdout: Any = lib.field(default = sys.stdout)
    groupname: str = lib.field(default = 'C1')
    topgroup: str = lib.field(default = 'C1')
    symm_orb: Optional[list] = lib.field(default = None)
    irrep_id: Optional[list] = lib.field(default = None)
    irrep_name: Optional[list] = lib.field(default = None)
    _symm_orig: Optional[numpy.ndarray] = lib.field(default = None)
    _symm_axes: Optional[numpy.ndarray] = lib.field(default = None)
    _nelectron: Optional[int] = lib.field(default = None)
    _nao: Optional[int] = lib.field(default = None)
    _enuc: Optional[float] = lib.field(default = None)
    _atom: list = lib.field(default_factory = list)
    _basis: dict = lib.field(default_factory = dict)
    _ecp: dict = lib.field(default_factory = dict)
    _built: bool = lib.field(default = False)
    _pseudo: dict = lib.field(default_factory = dict)

    def __post_init__(self):
        self._keys = set(self.__dict__.keys())

    def build(self, *args, **kwargs):
        gto.Mole.build(self, *args, **kwargs)
       
        self.coords = jnp.asarray(self.atom_coords())

    energy_nuc = energy_nuc

    def intor(self, intor, comp=None, hermi=0, aosym='s1', out=None,
              shls_slice=None):
        has_grad = ["int1e_ovlp", 
                    "int1e_kin",
                    "int1e_nuc",
                    "ECPscalar",
                    "int2e",]
        if not intor in has_grad:
            return gto.Mole.intor(self, intor, comp=comp, hermi=hermi, aosym=aosym, out=out, shls_slice=shls_slice)
        else:
            return moleintor.getints(self, intor)
