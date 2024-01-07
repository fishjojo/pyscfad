from jax import vmap
from pyscf import numpy as np
from pyscf import gto
from pyscf.lib import param
from pyscfad import util
from pyscfad import config
from pyscfad.lib import custom_jvp
from pyscfad.gto import moleintor
from pyscfad.gto.eval_gto import eval_gto
from ._mole_helper import setup_exp, setup_ctr_coeff

Traced_Attributes = ['coords', 'exp', 'ctr_coeff', 'r0']

def energy_nuc(mol, charges=None, **kwargs):
    if charges is None:
        charges = mol.atom_charges()
    if len(charges) <= 1:
        return 0.0
    r = distance_matrix(mol.atom_coords())
    enuc = np.einsum('i,ij,j->', charges, 1./r, charges) * .5
    return enuc

@custom_jvp
def distance_matrix(coords):
    r  = np.linalg.norm(coords[:,None,:] - coords[None,:,:], axis=2)
    r += np.eye(r.shape[-1]) * 1e200
    return r

@distance_matrix.defjvp
def distance_matrix_jvp(primals, tangents):
    coords, = primals
    coords_t, = tangents
    rnorm = primal_out = distance_matrix(coords)
    def body(r1, r2, rnorm, coords_t):
        r = r1 - r2
        jvp = np.dot(r / rnorm[:,None], coords_t)
        return jvp
    tangent_out = vmap(body, (0,None,0,0))(coords, coords, rnorm, coords_t)
    tangent_out += tangent_out.T
    return primal_out, tangent_out

intor_cross = moleintor.intor_cross

def nao_nr_range(mol, bas_id0, bas_id1):
    from pyscf.gto.moleintor import make_loc
    if mol.cart:
        key = 'cart'
    else:
        key = 'sph'
    ao_loc = make_loc(mol._bas[:bas_id1], key)
    nao_id0 = ao_loc[bas_id0]
    nao_id1 = ao_loc[-1]
    return nao_id0, nao_id1

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
        super().__init__(**kwargs)

    def atom_coords(self, unit='Bohr'):
        if self.coords is None:
            return super().atom_coords(unit)
        else:
            if unit[:3].upper() == 'ANG':
                return self.coords * param.BOHR
            else:
                return self.coords

    def build(self, *args, **kwargs):
        trace_coords = kwargs.pop('trace_coords', True)
        trace_exp = kwargs.pop('trace_exp', True)
        trace_ctr_coeff = kwargs.pop('trace_ctr_coeff', True)
        trace_r0 = kwargs.pop('trace_r0', False)

        super().build(*args, **kwargs)

        if trace_coords:
            self.coords = np.asarray(self.atom_coords())
        if trace_exp:
            self.exp = np.asarray(setup_exp(self)[0])
        if trace_ctr_coeff:
            self.ctr_coeff = np.asarray(setup_ctr_coeff(self)[0])
        if trace_r0:
            raise NotImplementedError

    energy_nuc = energy_nuc
    eval_ao = eval_gto = eval_gto

    def intor(self, intor, comp=None, hermi=0, aosym='s1', out=None,
              shls_slice=None, grids=None):
        if (self.coords is None and self.exp is None
                and self.ctr_coeff is None and self.r0 is None):
            return super().intor(intor, comp=comp, hermi=hermi,
                                 aosym=aosym, out=out, shls_slice=shls_slice,
                                 grids=grids)
        else:
            intor = self._add_suffix(intor)
            if config.moleintor_opt:
                from pyscfad.gto import moleintor_opt
                return moleintor_opt.getints(
                            self, intor, shls_slice=shls_slice,
                            comp=comp, hermi=hermi, aosym=aosym,
                            out=out, grids=grids)

            return moleintor.getints(self, intor, shls_slice=shls_slice,
                                     comp=comp, hermi=hermi, aosym=aosym,
                                     out=out, grids=grids)
