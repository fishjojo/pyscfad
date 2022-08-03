import warnings
from functools import partial
import numpy
#from scipy.special import erfc
from jax import custom_jvp
from jax.scipy.special import erfc
from pyscf import __config__
from pyscf.gto.mole import PTR_COORD
from pyscf.gto.moleintor import _get_intor_and_comp
from pyscf.pbc.gto import cell as pyscf_cell
from pyscfad import lib
from pyscfad import util
from pyscfad.lib import numpy as np
from pyscfad.lib import stop_grad
from pyscfad.gto import mole
from pyscfad.gto._mole_helper import setup_exp, setup_ctr_coeff
from pyscfad.pbc.gto import _pbcintor
from pyscfad.pbc.gto.eval_gto import eval_gto as pbc_eval_gto

def get_SI(cell, Gv=None):
    coords = cell.atom_coords()
    ngrids = numpy.prod(cell.mesh)
    if Gv is None or Gv.shape[0] == ngrids:
        Gv = cell.get_Gv()
    GvT = Gv.T
    SI = np.exp(-1j*np.dot(coords, GvT))
    return SI

energy_nuc = pyscf_cell.ewald

def shift_bas_center(cell0, r):
    cell = cell0.copy()
    cell.coords = cell0.atom_coords() + r[None,:]

    ptr = cell._atm[:,PTR_COORD]
    idx = numpy.vstack((ptr, ptr+1, ptr+2)).T.flatten()
    numpy.put(cell._env, idx, stop_grad(cell.coords).flatten())
    return cell

def intor_cross(intor, cell1, cell2, comp=None, hermi=0, kpts=None, kpt=None,
                shls_slice=None, **kwargs):
    intor, comp = _get_intor_and_comp(cell1._add_suffix(intor), comp)

    if kpts is None:
        if kpt is not None:
            kpts_lst = np.reshape(kpt, (1,3))
        else:
            kpts_lst = np.zeros((1,3))
    else:
        kpts_lst = np.reshape(kpts, (-1,3))

    Ls = cell2.get_lattice_Ls(rcut=max(cell1.rcut, cell2.rcut))
    expkL = np.exp(1j*np.dot(kpts_lst, Ls.T))

    ni = cell1.nao
    nj = cell2.nao
    nL = len(Ls)
    ints = []
    for i in range(nL):
        shifted_cell = shift_bas_center(cell2, Ls[i])
        ints.append(mole.intor_cross(intor, cell1, shifted_cell, comp=comp))
    ints = np.asarray(ints)

    if comp == 1:
        out = np.einsum('kl,lij->kij', expkL, ints)
    else:
        out = np.einsum('kl,lcij->kcij', expkL, ints)

    if kpts is None or np.shape(kpts) == (3,):  # A single k-point
        out = out[0]
    return out

def pbc_intor(cell, intor, comp=None, hermi=0, kpts=None, kpt=None,
              shls_slice=None, **kwargs):
    if kwargs:
        warnings.warn("Keyword arguments %s are ignored" % list(kwargs.keys()))

    if cell.abc is None:
        res = _pbcintor._pbc_intor(cell, intor, comp=comp, hermi=hermi, kpts=kpts,
                                   kpt=kpt, shls_slice=shls_slice)
    else:
        res = intor_cross(intor, cell, cell, comp=comp, hermi=hermi, kpts=kpts,
                          kpt=kpt, shls_slice=shls_slice, **kwargs)
    return res

@util.pytree_node(mole.Traced_Attributes+['abc'])
class Cell(mole.Mole, pyscf_cell.Cell):
    def __init__(self, **kwargs):
        mole.Mole.__init__(self, **kwargs)
        self.a = None # lattice vectors, (a1,a2,a3)
        self.abc = None # traced lattice vectors
        # if set, defines a spherical cutoff
        # of fourier components, with .5 * G**2 < ke_cutoff
        self.ke_cutoff = None

        self.pseudo = None
        self.dimension = 3
        # TODO: Simple hack for now; the implementation of ewald depends on the
        #       density-fitting class.  This determines how the ewald produces
        #       its energy.
        self.low_dim_ft_type = None

##################################################
# These attributes are initialized by build function if not given
        self.mesh = None
        self.rcut = None

##################################################
# don't modify the following variables, they are not input arguments
        keys = ('precision', 'exp_to_discard')
        self._keys = self._keys.union(self.__dict__).union(keys)
        self.__dict__.update(kwargs)


    def build(self, *args, **kwargs):
        trace_coords = kwargs.pop("trace_coords", True)
        trace_exp = kwargs.pop("trace_exp", False)
        trace_ctr_coeff = kwargs.pop("trace_ctr_coeff", False)
        trace_r0 = kwargs.pop("trace_r0", False)
        trace_lattice_vectors = kwargs.pop("trace_lattice_vectors", False)

        pyscf_cell.Cell.build(self, *args, **kwargs)

        if trace_coords:
            self.coords = np.asarray(self.atom_coords())
        if trace_exp:
            self.exp, _, _ = setup_exp(self)
        if trace_ctr_coeff:
            self.ctr_coeff, _, _ = setup_ctr_coeff(self)
        if trace_r0:
            pass
        if trace_lattice_vectors:
            self.abc = np.asarray(self.lattice_vectors())

    def lattice_vectors(self):
        if self.abc is None:
            return pyscf_cell.Cell.lattice_vectors(self)
        else:
            return self.abc

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
    pbc_intor = pbc_intor
    get_SI = get_SI
    energy_nuc = energy_nuc
