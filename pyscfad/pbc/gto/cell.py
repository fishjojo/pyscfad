import warnings
from typing import Optional, Any
import numpy
from pyscf import __config__
from pyscf.pbc.gto import cell
from pyscfad import lib
from pyscfad.lib import numpy as jnp
from pyscfad.gto import mole
from pyscfad.gto._mole_helper import setup_exp, setup_ctr_coeff
from pyscfad.pbc.gto import _pbcintor

def pbc_intor(mol, intor, comp=None, hermi=0, kpts=None, kpt=None,
              shls_slice=None, **kwargs):
    if kwargs:
        warnings.warn("Keyword arguments %s are ignored" % list(kwargs.keys()))
    res = _pbcintor._pbc_intor(mol, intor, comp=comp, hermi=hermi, kpts=kpts, 
                               kpt=kpt, shls_slice=shls_slice)
    return res

def get_SI(cell, Gv=None):
    coords = cell.coords
    if coords is None:
        coords = cell.atom_coords()
    ngrids = numpy.prod(cell.mesh)
    if Gv is None or Gv.shape[0] == ngrids:
        Gv = cell.get_Gv()
    GvT = Gv.T
    SI = jnp.exp(-1j*jnp.dot(coords, GvT))
    return SI


@lib.dataclass
class Cell(mole.Mole, cell.Cell):
    precision: float = getattr(__config__, 'pbc_gto_cell_Cell_precision', 1e-8)
    exp_to_discard: Optional[float] = getattr(__config__, 'pbc_gto_cell_Cell_exp_to_discard', None)

    a: Any = None
    ke_cutoff: Optional[float] = None
    pseudo: Any = None
    dimension: int = 3
    low_dim_ft_type: Optional[str] = None

    _mesh: Any = None
    _mesh_from_build: bool = True
    _rcut: Optional[float] = None

    @property
    def mesh(self):
        return self._mesh
    @mesh.setter
    def mesh(self, x):
        self._mesh = x
        self._mesh_from_build = False

    @property
    def rcut(self):
        return self._rcut
    @rcut.setter
    def rcut(self, x):
        self._rcut = x
        self._rcut_from_build = False

    def __post_init__(self):
        mole.Mole.__post_init__(self)

    def build(self, *args, **kwargs):
        trace_coords = kwargs.pop("trace_coords", False)
        trace_exp = kwargs.pop("trace_exp", False)
        trace_ctr_coeff = kwargs.pop("trace_ctr_coeff", False)
        trace_r0 = kwargs.pop("trace_r0", False)

        cell.Cell.build(self, *args, **kwargs)

        if trace_coords:
            self.coords = jnp.asarray(self.atom_coords())
        if trace_exp:
            self.exp, _, _ = setup_exp(self)
        if trace_ctr_coeff:
            self.ctr_coeff, _, _ = setup_ctr_coeff(self)
        if trace_r0:
            pass

    pbc_intor = pbc_intor
    get_SI = get_SI
