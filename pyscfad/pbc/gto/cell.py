# Copyright 2021-2025 Xing Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
import numpy
from jax.scipy.special import erf, erfc
from pyscf import __config__
from pyscf.lib import with_doc
from pyscf.gto.mole import PTR_COORD, ANG_OF
from pyscf.gto.moleintor import _get_intor_and_comp
from pyscf.pbc.gto import cell as pyscf_cell

from pyscfad import numpy as np
from pyscfad import ops
from pyscfad.ops import stop_grad
from pyscfad.lib import logger
from pyscfad.lib import cartesian_prod
from pyscfad.gto import mole
from pyscfad.gto._mole_helper import setup_exp, setup_ctr_coeff
from pyscfad.pbc.gto import _pbcintor
from pyscfad.pbc.gto.eval_gto import eval_gto as pbc_eval_gto
from pyscfad.pbc import tools as pbctools

@with_doc(pyscf_cell.get_Gv.__doc__)
def get_Gv(cell, mesh=None):
    return get_Gv_weights(cell, mesh)[0]

@with_doc(pyscf_cell.get_Gv_weights.__doc__)
def get_Gv_weights(cell, mesh=None):
    if mesh is None:
        mesh = cell.mesh

    # Default, the 3D uniform grids
    rx = np.fft.fftfreq(mesh[0], 1./mesh[0])
    ry = np.fft.fftfreq(mesh[1], 1./mesh[1])
    rz = np.fft.fftfreq(mesh[2], 1./mesh[2])
    b = cell.reciprocal_vectors()
    weights = abs(np.linalg.det(b))

    if (cell.dimension < 2 or
        (cell.dimension == 2 and cell.low_dim_ft_type == 'inf_vacuum')):
        raise NotImplementedError

    Gvbase = (rx, ry, rz)
    Gv = cartesian_prod(Gvbase) @ b
    Gv = Gv.reshape(-1, 3)

    # 1/cell.vol == det(b)/(2pi)^3
    weights *= 1/(2*np.pi)**3
    return Gv, Gvbase, weights

@with_doc(pyscf_cell.get_SI.__doc__)
def get_SI(cell, Gv=None, mesh=None, atmlst=None):
    coords = cell.atom_coords()
    if atmlst is not None:
        coords = coords[numpy.asarray(atmlst)]
    if Gv is None:
        if mesh is None:
            mesh = cell.mesh
        basex, basey, basez = cell.get_Gv_weights(mesh)[1]
        b = cell.reciprocal_vectors()
        rb = coords @ b.T
        SIx = np.exp(-1j*np.einsum('z,g->zg', rb[:,0], basex))
        SIy = np.exp(-1j*np.einsum('z,g->zg', rb[:,1], basey))
        SIz = np.exp(-1j*np.einsum('z,g->zg', rb[:,2], basez))
        SI = SIx[:,:,None,None] * SIy[:,None,:,None] * SIz[:,None,None,:]
        natm = coords.shape[0]
        SI = SI.reshape(natm, -1)
    else:
        SI = np.exp(-1j * (coords @ Gv.T))
    return SI

@with_doc(pyscf_cell.get_uniform_grids.__doc__)
def get_uniform_grids(cell, mesh=None, wrap_around=True):
    if mesh is None:
        mesh = cell.mesh

    a = cell.lattice_vectors()
    if wrap_around:
        qv = cartesian_prod([np.fft.fftfreq(x) for x in mesh])
        coords = qv @ a
    else:
        mesh = np.asarray(mesh, float)
        qv = cartesian_prod([np.arange(x) for x in mesh])
        a_frac = (1./mesh)[:,None] * a
        coords = qv @ a_frac
    return coords

gen_uniform_grids = get_uniform_grids

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
        warnings.warn(f'Keyword arguments {list(kwargs.keys())} are ignored')

    if cell.abc is None:
        res = _pbcintor._pbc_intor(cell, intor, comp=comp, hermi=hermi, kpts=kpts,
                                   kpt=kpt, shls_slice=shls_slice)
    else:
        res = intor_cross(intor, cell, cell, comp=comp, hermi=hermi, kpts=kpts,
                          kpt=kpt, shls_slice=shls_slice, **kwargs)
    return res

@with_doc(pyscf_cell.get_ewald_params.__doc__)
def get_ewald_params(cell, precision=None, mesh=None):
    if cell.natm == 0:
        return 0, 0

    if precision is None:
        precision = cell.precision

    if (cell.dimension < 2 or
          (cell.dimension == 2 and cell.low_dim_ft_type == 'inf_vacuum')):
        ew_cut = cell.rcut
        ew_eta = numpy.sqrt(max(numpy.log(4*numpy.pi*ew_cut**2/precision)/ew_cut**2, .1))
    elif cell.dimension == 2:
        a = ops.to_numpy(cell.lattice_vectors())
        ew_cut = a[2,2] / 2
        # ewovrl ~ erfc(eta*rcut) / rcut ~ e^{(-eta**2 rcut*2)} < precision
        log_precision = numpy.log(precision / (cell.atom_charges().sum()*16*numpy.pi**2))
        ew_eta = (-log_precision)**.5 / ew_cut
    else:  # dimension == 3
        ew_eta = 1./ops.to_numpy(cell.vol)**(1./6)
        ew_cut = _estimate_rcut(ew_eta**2, 0, 1., precision)
    return ew_eta, ew_cut

@with_doc(pyscf_cell.ewald.__doc__)
def ewald(cell, ew_eta=None, ew_cut=None):
    if cell.a is None:
        return mole.energy_nuc(cell)

    if cell.natm == 0:
        return 0

    chargs = cell.atom_charges()

    if ew_eta is None or ew_cut is None:
        ew_eta, ew_cut = cell.get_ewald_params()
    log_precision = numpy.log(cell.precision / (chargs.sum()*16*numpy.pi**2))
    ke_cutoff = -2*ew_eta**2*log_precision
    mesh = cell.cutoff_to_mesh(ke_cutoff)
    logger.debug1(cell, 'mesh for ewald %s', mesh)

    coords = cell.atom_coords()
    Lall = cell.get_lattice_Ls(rcut=ew_cut)

    rLij = coords[:,None,:] - coords[None,:,:] + Lall[:,None,None,:]
    r2 = np.einsum('Lijx,Lijx->Lij', rLij, rLij)
    # avoid gradient divergence
    r = np.sqrt(np.where(r2>1e-12, r2, 0))
    r = np.where(r>1e-6, r, np.inf)
    ewovrl = .5 * np.einsum('i,j,Lij->', chargs, chargs, erfc(ew_eta * r) / r)

    ewself  = -.5 * numpy.dot(chargs,chargs) * 2 * ew_eta / numpy.sqrt(numpy.pi)
    if cell.dimension == 3:
        ewself += -.5 * numpy.sum(chargs)**2 * numpy.pi/(ew_eta**2 * cell.vol)

    if cell.dimension != 2 or cell.low_dim_ft_type == 'inf_vacuum':
        Gv, Gvbase, weights = cell.get_Gv_weights(mesh)
        absG2 = np.einsum('gi,gi->g', Gv, Gv)
        absG2 = np.where(absG2==0, np.inf, absG2)

        coulG = 4*np.pi / absG2
        coulG *= weights

        #:ZSI = np.einsum('i,ij->j', chargs, cell.get_SI(Gv))
        basex, basey, basez = Gvbase
        b = cell.reciprocal_vectors()
        rb = np.dot(coords, b.T)
        SIx = np.exp(-1j*np.einsum('z,g->zg', rb[:,0], basex))
        SIy = np.exp(-1j*np.einsum('z,g->zg', rb[:,1], basey))
        SIz = np.exp(-1j*np.einsum('z,g->zg', rb[:,2], basez))
        ZSI = np.einsum('i,ix,iy,iz->xyz', chargs, SIx, SIy, SIz).ravel()
        ZexpG2 = ZSI * np.exp(-absG2/(4*ew_eta**2))
        ewg = .5 * np.einsum('i,i,i', ZSI.conj(), ZexpG2, coulG).real

    elif cell.dimension == 2:  # Truncated Coulomb
        # The following 2D ewald summation is taken from:
        # R. Sundararaman and T. Arias PRB 87, 2013
        def fn(eta,Gnorm,z):
            Gnorm_z = Gnorm*z
            large_idx = ops.index[Gnorm_z > 20.0]
            ok_idx = ops.index[Gnorm_z <= 20.0]
            ret = np.zeros_like(Gnorm_z)
            x = Gnorm/2./eta + eta*z
            with numpy.errstate(over='ignore'):
                erfcx = erfc(x)
                #:ret[~large_idx] = np.exp(Gnorm_z[~large_idx]) * erfcx[~large_idx]
                ret = ops.index_update(ret, ok_idx,
                                       np.exp(Gnorm_z[ok_idx]) * erfcx[ok_idx])
                #:ret[ large_idx] = np.exp((Gnorm*z-x**2)[large_idx]) * erfcx[large_idx]
                ret = ops.index_update(ret, large_idx,
                                       np.exp((Gnorm*z-x**2)[large_idx]) * erfcx[large_idx])
            return ret
        def gn(eta,Gnorm,z):
            return np.pi/Gnorm*(fn(eta,Gnorm,z) + fn(eta,Gnorm,-z))
        def gn0(eta,z):
            return -2*np.pi*(z*erf(eta*z) + np.exp(-(eta*z)**2)/eta/numpy.sqrt(numpy.pi))
        b = cell.reciprocal_vectors()
        inv_area = np.linalg.norm(np.cross(b[0], b[1]))/(2*np.pi)**2
        # Perform the reciprocal space summation over  all reciprocal vectors
        # within the x,y plane.
        Gv, Gvbase, weights = cell.get_Gv_weights(mesh)
        absG2 = np.einsum('gi,gi->g', Gv, Gv)
        planarG2_idx = np.logical_and(Gv[:,2] == 0, absG2 > 0)
        Gv = Gv[planarG2_idx]
        absG2 = absG2[planarG2_idx]
        absG = absG2**(0.5)
        # Performing the G != 0 summation.
        rij = coords[:,None,:] - coords[None,:,:]
        Gdotr = np.einsum('ijx,gx->ijg', rij, Gv)
        ewg = np.einsum('i,j,ijg,ijg->', chargs, chargs, np.cos(Gdotr),
                        gn(ew_eta,absG,rij[:,:,2:3]))
        # Performing the G == 0 summation.
        ewg += np.einsum('i,j,ij->', chargs, chargs, gn0(ew_eta,rij[:,:,2]))
        ewg *= inv_area*0.5
    else:
        logger.warn(cell, 'No method for PBC dimension %s, dim-type %s.'
                    '  cell.low_dim_ft_type="inf_vacuum"  should be set.',
                    cell.dimension, cell.low_dim_ft_type)
        raise NotImplementedError

    logger.debug(cell, 'Ewald components = %.15g, %.15g, %.15g', ewovrl, ewself, ewg)
    return ewovrl + ewself + ewg

energy_nuc = ewald


def _estimate_rcut(alpha, l, c, precision=1e-8):
    r"""Similar as :func:`pyscf.pbc.gto.cell._estimate_rcut`,
    but with cutoff radius estimated based on

    .. math::
        \int \phi(0) \phi(r_0) < \epsilon
    """
    theta = alpha * .5
    a1 = (alpha * 2)**-.5
    norm_ang = (2*l+1)/(4*numpy.pi)
    fac = norm_ang * (.5*numpy.pi/alpha)**1.5 / precision
    r0 = 20
    r0 = (numpy.log(fac * (r0*.5+a1)**(2*l)) / theta)**.5
    r0 = (numpy.log(fac * (r0*.5+a1)**(2*l)) / theta)**.5
    return r0

def bas_rcut(cell, bas_id, precision=None):
    """Same as :func:`pyscf.pbc.gto.cell.bas_rcut`,
    but gives slightly different cutoff radius.
    """
    if precision is None:
        precision = cell.precision
    l = cell.bas_angular(bas_id)
    es = cell.bas_exp(bas_id)
    cs = abs(cell._libcint_ctr_coeff(bas_id)).max(axis=1)
    rcut = _estimate_rcut(es, l, cs, precision)
    return rcut.max()

def estimate_rcut(cell, precision=None):
    """Same as :func:`pyscf.pbc.gto.cell.estimate_rcut`,
    but gives slightly different cutoff radius.
    """
    if cell.nbas == 0:
        return 0.01
    if precision is None:
        precision = cell.precision
    if cell.use_loose_rcut:
        return cell.rcut_by_shells(precision).max()

    exps, cs = pyscf_cell._extract_pgto_params(cell, 'min')
    ls = cell._bas[:,ANG_OF]
    rcut = _estimate_rcut(exps, ls, cs, precision)
    return rcut.max()

# FIXME monkey patch
pyscf_cell.estimate_rcut = estimate_rcut

class Cell(mole.Mole, pyscf_cell.Cell):
    """Subclass of :class:`pyscf.pbc.gto.Cell` with traceable attributes.

    Attributes
    ----------
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
    abc : array
        Lattice vectors.
    """
    _dynamic_attr = _keys = {'abc'}

    def __init__(self, **kwargs):
        self.coords = None
        self.exp = None
        self.ctr_coeff = None
        self.r0 = None
        self.abc = None
        pyscf_cell.Cell.__init__(self, **kwargs)

    def build(self, *args, **kwargs):
        trace_coords = kwargs.pop('trace_coords', True)
        trace_exp = kwargs.pop('trace_exp', False)
        trace_ctr_coeff = kwargs.pop('trace_ctr_coeff', False)
        trace_r0 = kwargs.pop('trace_r0', False)
        trace_lattice_vectors = kwargs.pop('trace_lattice_vectors', False)

        pyscf_cell.Cell.build(self, *args, **kwargs)

        if trace_coords:
            self.coords = np.asarray(self.atom_coords())
        if trace_exp:
            self.exp, _, _ = setup_exp(self)
        if trace_ctr_coeff:
            self.ctr_coeff, _, _ = setup_ctr_coeff(self)
        if trace_r0:
            raise NotImplementedError
        if trace_lattice_vectors:
            self.abc = np.asarray(self.lattice_vectors())

    @property
    def vol(self):
        return abs(np.linalg.det(self.lattice_vectors()))

    def lattice_vectors(self):
        if self.abc is None:
            return pyscf_cell.Cell.lattice_vectors(self)
        else:
            return self.abc

    @with_doc(pyscf_cell.Cell.get_scaled_atom_coords.__doc__)
    def get_scaled_atom_coords(self, a=None):
        if a is None:
            a = self.lattice_vectors()
        return np.dot(self.atom_coords(), np.linalg.inv(a))

    @with_doc(pyscf_cell.Cell.reciprocal_vectors.__doc__)
    def reciprocal_vectors(self, norm_to=2*np.pi):
        a = self.lattice_vectors()
        if self.dimension == 1:
            assert(abs(a[0] @ a[1]) < 1e-9 and
                   abs(a[0] @ a[2]) < 1e-9 and
                   abs(a[1] @ a[2]) < 1e-9)
        elif self.dimension == 2:
            assert(abs(a[0] @ a[2]) < 1e-9 and
                   abs(a[1] @ a[2]) < 1e-9)
        b = np.linalg.inv(a.T)
        return norm_to * b

    @with_doc(pyscf_cell.Cell.get_abs_kpts.__doc__)
    def get_abs_kpts(self, scaled_kpts):
        return np.dot(scaled_kpts, self.reciprocal_vectors())

    @with_doc(pyscf_cell.Cell.get_scaled_kpts.__doc__)
    def get_scaled_kpts(self, abs_kpts, kpts_in_ibz=True):
        from pyscf.pbc.lib.kpts import KPoints
        if isinstance(abs_kpts, KPoints):
            if kpts_in_ibz:
                return abs_kpts.kpts_scaled_ibz
            else:
                return abs_kpts.kpts_scaled
        return 1./(2*np.pi)*np.dot(abs_kpts, self.lattice_vectors().T)

    @with_doc(pyscf_cell.Cell.cutoff_to_mesh.__doc__)
    def cutoff_to_mesh(self, ke_cutoff):
        a = self.lattice_vectors()
        dim = self.dimension
        mesh = pbctools.cutoff_to_mesh(a, ke_cutoff)
        if dim < 2 or (dim == 2 and self.low_dim_ft_type == 'inf_vacuum'):
            mesh[dim:] = self.mesh[dim:]
        return mesh

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

    def to_pyscf(self):
        cell = self.view(pyscf_cell.Cell)
        del cell.coords
        del cell.exp
        del cell.ctr_coeff
        del cell.r0
        del cell.abc
        return cell

    bas_rcut = bas_rcut
    eval_ao = eval_gto
    pbc_intor = pbc_intor
    get_Gv = get_Gv
    get_Gv_weights = get_Gv_weights
    get_SI = get_SI
    gen_uniform_grids = get_uniform_grids = get_uniform_grids
    get_ewald_params = get_ewald_params
    ewald = ewald
    energy_nuc = ewald
    get_lattice_Ls = pbctools.get_lattice_Ls
