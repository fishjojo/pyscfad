import warnings
from functools import partial
from typing import Optional, Any
import numpy
from scipy.special import erfc
from jax import custom_jvp
from pyscf import __config__
from pyscf.pbc.gto import cell as pyscf_cell
from pyscfad import lib
from pyscfad.lib import numpy as jnp
from pyscfad.gto import mole
from pyscfad.gto._mole_helper import setup_exp, setup_ctr_coeff
from pyscfad.pbc.gto import _pbcintor
from pyscfad.pbc.gto.eval_gto import eval_gto as pbc_eval_gto

def pbc_intor(mol, intor, comp=None, hermi=0, kpts=None, kpt=None,
              shls_slice=None, **kwargs):
    if kwargs:
        warnings.warn("Keyword arguments %s are ignored" % list(kwargs.keys()))
    res = _pbcintor._pbc_intor(mol, intor, comp=comp, hermi=hermi, kpts=kpts, 
                               kpt=kpt, shls_slice=shls_slice)
    return res

def get_SI(cell, Gv=None):
    coords = cell.atom_coords()
    ngrids = numpy.prod(cell.mesh)
    if Gv is None or Gv.shape[0] == ngrids:
        Gv = cell.get_Gv()
    GvT = Gv.T
    SI = jnp.exp(-1j*jnp.dot(coords, GvT))
    return SI

energy_nuc = pyscf_cell.ewald

def _ewald_sr(cell, coords=None, charges=None, ew_eta=None, ew_cut=None, Lall=None):
    return _ewald_sr_wrap(cell, coords, charges, ew_eta, ew_cut, Lall)

@partial(custom_jvp, nondiff_argnums=tuple(range(2,6)))
def _ewald_sr_wrap(cell, coords, charges, ew_eta, ew_cut, Lall):
    return pyscf_cell._ewald_sr(cell, coords, charges, ew_eta, ew_cut, Lall)

@_ewald_sr_wrap.defjvp
def _ewald_sr_jvp(charges, ew_eta, ew_cut, Lall, primals, tangents):
    cell, coords, = primals
    cell_t, coords_t, = tangents

    primal_out = _ewald_sr_wrap(cell, coords, charges, ew_eta, ew_cut, Lall)

    if coords is None:
        coords = cell.coords
        coords_t = cell_t.coords_t
 
    if charges is None:
        charges = cell.atom_charges()
    if ew_eta is None:
        ew_eta = cell.get_ewald_params()[0]
    if ew_cut is None:
        ew_cut = cell.get_ewald_params()[1]
    if Lall is None:
        Lall = cell.get_lattice_Ls(rcut=ew_cut)

    natm = cell.natm
    grad = numpy.zeros((natm,3), dtype=numpy.double)
    coords = numpy.asarray(coords)
    rLij = coords[:,None,:] - coords[None,:,:] + Lall[:,None,None,:]
    r2 = numpy.einsum('Lijx,Lijx->Lij', rLij, rLij)
    r = numpy.sqrt(r2)
    r = numpy.where(r > 1e-16, r, 1e200)
    r2 = numpy.where(r2 > 1e-16, r2, 1e200)

    tmp = - erfc(ew_eta *  r) / r2 
    tmp += - 2. * ew_eta / (numpy.sqrt(numpy.pi) * r) * numpy.exp(-r2 * ew_eta**2)
    r2 = None
    for i in range(natm):
        dr = rLij[:,i] / r[:,i,:,None]
        grad[i] += charges[i] * numpy.einsum('j,Lj,Ljx->', charges, tmp[:,i], dr)
    tmp = rLij = r = None
    tangent_out = jnp.einsum('nx,nx->', grad, coords_t)
    return primal_out, tangent_out

@lib.dataclass
class Cell(mole.Mole, pyscf_cell.Cell):
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
        trace_coords = kwargs.pop("trace_coords", True)
        trace_exp = kwargs.pop("trace_exp", False)
        trace_ctr_coeff = kwargs.pop("trace_ctr_coeff", False)
        trace_r0 = kwargs.pop("trace_r0", False)

        pyscf_cell.Cell.build(self, *args, **kwargs)

        if trace_coords:
            self.coords = jnp.asarray(self.atom_coords())
        if trace_exp:
            self.exp, _, _ = setup_exp(self)
        if trace_ctr_coeff:
            self.ctr_coeff, _, _ = setup_ctr_coeff(self)
        if trace_r0:
            pass

    def pbc_eval_gto(self, eval_name, coords, comp=None, kpts=None, kpt=None,
                     shls_slice=None, non0tab=None, ao_loc=None, out=None):
        return pbc_eval_gto(self, eval_name, coords, comp, kpts, kpt,
                            shls_slice, non0tab, ao_loc, out)
    pbc_eval_ao = pbc_eval_gto

    def eval_gto(self, eval_name, coords, comp=None, kpts=None, kpt=None,
                 shls_slice=None, non0tab=None, ao_loc=None, out=None):
        if eval_name[:3] == 'PBC':
            return self.pbc_eval_gto(eval_name, coords, comp, kpts, kpt,
                                     shls_slice, non0tab, ao_loc, out)
        else:
            return mole.eval_gto(self, eval_name, coords, comp,
                                 shls_slice, non0tab, ao_loc, out)
    eval_ao = eval_gto
    _ewald_sr = _ewald_sr
    pbc_intor = pbc_intor
    get_SI = get_SI
    energy_nuc = energy_nuc
