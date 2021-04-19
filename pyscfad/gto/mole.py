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
from pyscfad.gto.eval_gto import eval_gto
from ._mole_helper import setup_exp, setup_ctr_coeff

def energy_nuc(mol, charges=None, coords=None):
    if charges is None:
        charges = mol.atom_charges()
    if len(charges) <= 1:
        return 0
    rr = inter_distance(mol, coords)
    rr = ops.index_update(rr, jnp.diag_indices_from(rr), 1.e200)
    e = jnp.einsum('i,ij,j->', charges, 1./rr, charges) * .5
    return e

def inter_distance(mol, coords=None):
    if coords is None:
        coords = mol.coords
    if coords is None:
        coords = mol.atom_coords()
    rr = jnp.linalg.norm(coords.reshape(-1,1,3) - coords, axis=2)
    rr = ops.index_update(rr, jnp.diag_indices_from(rr), 0.)
    return rr

@lib.dataclass
class Mole(gto.Mole):
    # traced attributes
    # NOTE jax requires that at least one variable needs to be traced for AD
    coords: Optional[jnp.array] = lib.field(pytree_node=True, default=None)
    exp: Optional[jnp.array] = lib.field(pytree_node=True, default=None)
    ctr_coeff: Optional[jnp.array] = lib.field(pytree_node=True, default=None)
    r0: Optional[jnp.array] = lib.field(pytree_node=True, default=None)

    # attributes of the base class
    verbose: int = getattr(__config__, 'VERBOSE', logger.NOTE)
    unit: str = getattr(__config__, 'UNIT', 'angstrom')
    incore_anyway: bool = getattr(__config__, 'INCORE_ANYWAY', False)
    cart: bool = getattr(__config__, 'gto_mole_Mole_cart', False)

    # attributes of the base class object
    output: Optional[str] = None
    max_memory: int = param.MAX_MEMORY
    charge: int = 0
    spin: int = 0
    symmetry: bool = False
    symmetry_subgroup: Optional[str] = None
    cart: bool = False
    atom: Union[list,str] = lib.field(default_factory = list)
    basis: Union[dict,str] = 'sto-3g'
    nucmod: Union[dict,str] = lib.field(default_factory = dict)
    ecp: Union[dict,str] = lib.field(default_factory = dict)
    nucprop: dict = lib.field(default_factory = dict)

    # private attributes
    _atm: numpy.ndarray = numpy.zeros((0,6), dtype=numpy.int32)
    _bas: numpy.ndarray = numpy.zeros((0,8), dtype=numpy.int32)
    _env: numpy.ndarray = numpy.zeros(PTR_ENV_START)
    _ecpbas: numpy.ndarray = numpy.zeros((0,8), dtype=numpy.int32)

    stdout: Any = sys.stdout
    groupname: str = 'C1'
    topgroup: str = 'C1'
    symm_orb: Optional[list] = None
    irrep_id: Optional[list] = None
    irrep_name: Optional[list] = None
    _symm_orig: Optional[numpy.ndarray] = None
    _symm_axes: Optional[numpy.ndarray] = None
    _nelectron: Optional[int] = None
    _nao: Optional[int] = None
    _enuc: Optional[float] = None
    _atom: list = lib.field(default_factory = list)
    _basis: dict = lib.field(default_factory = dict)
    _ecp: dict = lib.field(default_factory = dict)
    _built: bool = False
    _pseudo: dict = lib.field(default_factory = dict)

    def __post_init__(self):
        self._keys = set(self.__dict__.keys())

    def build(self, *args, **kwargs):
        trace_coords = kwargs.pop("trace_coords", False)
        trace_exp = kwargs.pop("trace_exp", False)
        trace_ctr_coeff = kwargs.pop("trace_ctr_coeff", False)
        trace_r0 = kwargs.pop("trace_r0", False)

        gto.Mole.build(self, *args, **kwargs)

        if trace_coords:
            self.coords = jnp.asarray(self.atom_coords())
        if trace_exp:
            self.exp, _, _ = setup_exp(self)
        if trace_ctr_coeff:
            self.ctr_coeff, _, _ = setup_ctr_coeff(self)
        if trace_r0:
            pass

    energy_nuc = energy_nuc
    eval_ao = eval_gto = eval_gto

    def intor(self, intor, comp=None, hermi=0, aosym='s1', out=None,
              shls_slice=None):
        if (self.coords is None and self.exp is None
                and self.ctr_coeff is None and self.r0 is None):
            return gto.Mole.intor(self, intor, comp=comp, hermi=hermi,
                                  aosym=aosym, out=out, shls_slice=shls_slice)
        else:
            return moleintor.getints(self, intor, shls_slice,
                                     comp, hermi, aosym, out=None)
