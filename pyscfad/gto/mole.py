from jax import custom_jvp
from pyscf import __config__
from pyscf import gto
from pyscf.lib import logger, param
from pyscf.gto.mole import PTR_ENV_START
from pyscfad import util
from pyscfad.lib import numpy as np
from pyscfad.lib import ops
from pyscfad.gto import moleintor
from pyscfad.gto.eval_gto import eval_gto
from ._mole_helper import setup_exp, setup_ctr_coeff

Traced_Attributes = ['coords', 'exp', 'ctr_coeff', 'r0']

def energy_nuc(mol, charges=None, coords=None):
    if charges is None:
        charges = mol.atom_charges()
    if len(charges) <= 1:
        return 0
    rr = inter_distance(mol, coords)
    e = np.einsum('i,ij,j->', charges, 1./rr, charges) * .5
    return e

def inter_distance(mol, coords=None):
    coords = mol.atom_coords()
    return _rr(coords)

@custom_jvp
def _rr(coords):
    coords = np.asarray(coords)
    rr = np.linalg.norm(coords.reshape(-1,1,3) - coords, axis=2)
    #rr[numpy.diag_indices_from(rr)] = 1e200
    rr = ops.index_update(rr, np.diag_indices_from(rr), 1e200)
    return rr

@_rr.defjvp
def _rr_jvp(primals, tangents):
    coords, = primals
    coords_t, = tangents

    rnorm = primal_out = _rr(coords)

    r = coords.reshape(-1,1,3) - coords
    natm = coords.shape[0]
    #tangent_out = np.zeros_like(primal_out)
    grad = np.zeros((natm,natm,3), dtype=float)
    for i in range(natm):
        #tangent_out = ops.index_add(tangent_out, ops.index[i],
        #                            np.dot(r[i] / rnorm[i,:,None], coords_t[i]))
        #grad[i] += r[i] / rnorm[i,:,None]
        grad = ops.index_add(grad, ops.index[i], r[i] / rnorm[i,:,None])
    tangent_out = np.einsum("ijx,ix->ij", grad, coords_t)
    tangent_out += tangent_out.T
    return primal_out, tangent_out

@util.pytree_node(Traced_Attributes)
class Mole(gto.Mole):
    '''
    A subclass of :class:`pyscf.gto.Mole`, where the following
    attributes can be traced.

    Attributes:
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
    '''
    def __init__(self, **kwargs):
        self.coords = None
        self.exp = None
        self.ctr_coeff = None
        self.r0 = None
        gto.Mole.__init__(self, **kwargs)

    def atom_coords(self, unit='Bohr'):
        if self.coords is None:
            return gto.Mole.atom_coords(self, unit)
        else:
            if unit[:3].upper() == 'ANG':
                return self.coords * param.BOHR
            else:
                return self.coords

    def build(self, *args, **kwargs):
        trace_coords = kwargs.pop("trace_coords", True)
        trace_exp = kwargs.pop("trace_exp", True)
        trace_ctr_coeff = kwargs.pop("trace_ctr_coeff", True)
        trace_r0 = kwargs.pop("trace_r0", False)

        gto.Mole.build(self, *args, **kwargs)

        if trace_coords:
            self.coords = np.asarray(self.atom_coords())
        if trace_exp:
            self.exp, _, _ = setup_exp(self)
        if trace_ctr_coeff:
            self.ctr_coeff, _, _ = setup_ctr_coeff(self)
        if trace_r0:
            raise NotImplementedError

    energy_nuc = energy_nuc
    eval_ao = eval_gto = eval_gto

    def intor(self, intor, comp=None, hermi=0, aosym='s1', out=None,
              shls_slice=None, grids=None):
        if (self.coords is None and self.exp is None
                and self.ctr_coeff is None and self.r0 is None):
            return gto.Mole.intor(self, intor, comp=comp, hermi=hermi,
                                  aosym=aosym, out=out, shls_slice=shls_slice)
        else:
            return moleintor.getints(self, intor, shls_slice,
                                     comp, hermi, aosym, out=None)
