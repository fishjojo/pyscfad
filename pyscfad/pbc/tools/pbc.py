# Copyright 2021-2026 The PySCFAD Authors
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

from functools import wraps
import warnings
import numpy
from pyscf.pbc import tools as pyscf_pbctools
from pyscfad import numpy as np
from pyscfad import ops
from pyscfad.ops import stop_trace, stop_grad
from pyscfad.lib import logger, cartesian_prod

@wraps(pyscf_pbctools.fft)
def fft(f, mesh):
    if f.size == 0:
        return numpy.zeros_like(f)

    f3d = f.reshape(-1, *mesh)
    assert(f3d.shape[0] == 1 or f[0].size == f3d[0].size)
    g3d = np.fft.fftn(f3d, axes=(1,2,3))
    ngrids = numpy.prod(mesh)
    if f.ndim == 1 or (f.ndim == 3 and f.size == ngrids):
        return g3d.ravel()
    else:
        return g3d.reshape(-1, ngrids)

@wraps(pyscf_pbctools.ifft)
def ifft(g, mesh):
    if g.size == 0:
        return numpy.zeros_like(g)

    g3d = g.reshape(-1, *mesh)
    assert(g3d.shape[0] == 1 or g[0].size == g3d[0].size)
    f3d = np.fft.ifftn(g3d, axes=(1,2,3))
    ngrids = numpy.prod(mesh)
    if g.ndim == 1 or (g.ndim == 3 and g.size == ngrids):
        return f3d.ravel()
    else:
        return f3d.reshape(-1, ngrids)

@wraps(pyscf_pbctools.fftk)
def fftk(f, mesh, expmikr):
    return fft(f*expmikr, mesh)

@wraps(pyscf_pbctools.ifftk)
def ifftk(g, mesh, expikr):
    return ifft(g, mesh) * expikr

def get_lattice_Ls(cell, nimgs=None, rcut=None, dimension=None, discard=True):
    """Get the lattice translation vectors for lattice sum.

    Same as :func:`pyscf.pbc.tools.get_lattice_Ls`,
    but gives slightly fewer periodic images.

    Parameters
    ----------
    nimgs: int or array
        Number of periodic images in each dimension.
        Can be a number or a sequence of size 3.
        Default is ``None``, which means to use ``rcut`` to determine
        the number of periodic images.

    Notes
    -----
    This function is not jit compatible unless static ``nimgs`` is set.
    """
    if dimension is None:
        # For atoms near the boundary of the cell, it is necessary (even in low-
        # dimensional systems) to include lattice translations in all 3 dimensions.
        if cell.dimension < 2 or cell.low_dim_ft_type == 'inf_vacuum':
            dimension = cell.dimension
        else:
            dimension = 3

    if dimension == 0 or cell.natm == 0:
        return np.zeros((1, 3))

    def find_boundary(aR):
        r = np.linalg.qr(aR.T)[1]
        ub = (rcut + abs(r[2,3:]).sum()) / abs(r[2,2])
        return ub

    a = cell.lattice_vectors()

    if nimgs is not None:
        if np.isscalar(nimgs):
            bounds = np.repeat(nimgs, 3)
        else:
            assert len(nimgs) == 3
            bounds = np.asarray(nimgs, dtype=int)
    else:
        if rcut is None:
            rcut = cell.rcut

        _a = stop_grad(a)
        scaled_atom_coords = cell.get_scaled_atom_coords(_a)
        atom_boundary_max = scaled_atom_coords[:,:dimension].max(axis=0)
        atom_boundary_min = scaled_atom_coords[:,:dimension].min(axis=0)
        ovlp_penalty = np.maximum(abs(atom_boundary_max), abs(atom_boundary_min))

        xb = find_boundary(_a[np.array([1,2,0])])
        if dimension > 1:
            yb = find_boundary(_a[np.array([2,0,1])])
        else:
            yb = 0
        if dimension > 2:
            zb = find_boundary(a)
        else:
            zb = 0

        bounds = np.asarray([xb, yb, zb]) + ovlp_penalty
        bounds = np.ceil(bounds).astype(int)

    Ts = cartesian_prod((np.arange(-bounds[0], bounds[0]+1),
                         np.arange(-bounds[1], bounds[1]+1),
                         np.arange(-bounds[2], bounds[2]+1)))

    Ls = np.dot(Ts[:,:dimension], a[:dimension])

    if nimgs is None and discard:
        rcut_penalty = np.linalg.norm(np.dot(atom_boundary_max - atom_boundary_min, _a))
        Ls_mask = np.where(np.linalg.norm(Ls, axis=1) < rcut + rcut_penalty)[0]
        Ls = Ls[Ls_mask]
    return Ls


@wraps(pyscf_pbctools.get_coulG)
def get_coulG(cell, k=numpy.zeros(3), exx=False, mf=None, mesh=None, Gv=None,
              wrap_around=True, omega=None, **kwargs):
    exxdiv = exx
    if isinstance(exx, str):
        exxdiv = exx
    elif exx and mf is not None:
        exxdiv = mf.exxdiv

    if mesh is None:
        mesh = cell.mesh
    if 'gs' in kwargs:
        warnings.warn('cell.gs is deprecated.  It is replaced by cell.mesh,'
                      'the number of PWs (=2*gs+1) along each direction.')
        mesh = [2*n+1 for n in kwargs['gs']]
    if Gv is None:
        Gv = cell.get_Gv(mesh)

    if abs(k).sum() > 1e-9:
        kG = k + Gv
    else:
        kG = Gv

    equal2boundary = numpy.zeros(Gv.shape[0], dtype=bool)
    if wrap_around and abs(k).sum() > 1e-9:
        b = cell.reciprocal_vectors()
        box_edge = np.einsum('i,ij->ij', numpy.asarray(mesh)//2+0.5, b)
        assert (all(stop_trace(numpy.linalg.solve)(box_edge.T, k).round(9).astype(int)==0))
        reduced_coords = stop_trace(numpy.linalg.solve)(box_edge.T, kG.T).T.round(9)
        on_edge = reduced_coords.astype(int)
        if cell.dimension >= 1:
            equal2boundary |= reduced_coords[:,0] == 1
            equal2boundary |= reduced_coords[:,0] ==-1
            kG = ops.index_add(kG, ops.index[on_edge[:,0]== 1], -2 * box_edge[0])
            kG = ops.index_add(kG, ops.index[on_edge[:,0]==-1],  2 * box_edge[0])
        if cell.dimension >= 2:
            equal2boundary |= reduced_coords[:,1] == 1
            equal2boundary |= reduced_coords[:,1] ==-1
            kG = ops.index_add(kG, ops.index[on_edge[:,1]== 1], -2 * box_edge[1])
            kG = ops.index_add(kG, ops.index[on_edge[:,1]==-1],  2 * box_edge[1])
        if cell.dimension == 3:
            equal2boundary |= reduced_coords[:,2] == 1
            equal2boundary |= reduced_coords[:,2] ==-1
            kG = ops.index_add(kG, ops.index[on_edge[:,2]== 1], -2 * box_edge[2])
            kG = ops.index_add(kG, ops.index[on_edge[:,2]==-1],  2 * box_edge[2])

    absG2 = np.einsum('gi,gi->g', kG, kG)
    G0_idx = np.where(absG2==0)[0]
    absG2 = np.where(absG2!=0, absG2, 0)

    if getattr(mf, 'kpts', None) is not None:
        kpts = mf.kpts
    else:
        kpts = k.reshape(1,3)
    Nk = len(kpts)

    if exxdiv == 'vcut_sph':  # PRB 77 193110
        Rc = (3*Nk*cell.vol/(4*numpy.pi))**(1./3)
        with numpy.errstate(divide='ignore',invalid='ignore'):
            coulG = 4*numpy.pi/absG2*(1.0 - np.cos(np.sqrt(absG2)*Rc))
        if len(G0_idx) > 0:
            coulG = ops.index_update(coulG, ops.index[G0_idx], 4*numpy.pi*0.5*Rc**2)

        if cell.dimension < 3:
            raise NotImplementedError
    elif exxdiv == 'vcut_ws':  # PRB 87, 165122
        raise NotImplementedError
    else:
        # Ewald probe charge method to get the leading term of the finite size
        # error in exchange integrals

        if cell.dimension != 2 or cell.low_dim_ft_type == 'inf_vacuum':
            with numpy.errstate(divide='ignore'):
                coulG = 4*numpy.pi/absG2
            if len(G0_idx) > 0:
                coulG = ops.index_update(coulG, ops.index[G0_idx], 0)

        elif cell.dimension == 2:
            # The following 2D analytical fourier transform is taken from:
            # R. Sundararaman and T. Arias PRB 87, 2013
            b = cell.reciprocal_vectors()
            Ld2 = numpy.pi/np.linalg.norm(b[2])
            Gz = kG[:,2]
            Gp = np.linalg.norm(kG[:,:2], axis=1)
            weights = 1. - np.cos(Gz*Ld2) * np.exp(-Gp*Ld2)
            with numpy.errstate(divide='ignore', invalid='ignore'):
                coulG = weights*4*numpy.pi/absG2
            if len(G0_idx) > 0:
                coulG = ops.index_update(coulG, ops.index[G0_idx], -2*numpy.pi*Ld2**2)

        elif cell.dimension == 1:
            logger.warn(cell, 'No method for PBC dimension 1, dim-type %s.'
                        '  cell.low_dim_ft_type="inf_vacuum"  should be set.',
                        cell.low_dim_ft_type)
            raise NotImplementedError

        if cell.dimension > 0 and exxdiv == 'ewald' and len(G0_idx) > 0:
            coulG = ops.index_add(coulG, ops.index[G0_idx], Nk*cell.vol*madelung(cell, kpts))

    if equal2boundary is not None:
        coulG = ops.index_update(coulG, ops.index[equal2boundary], 0)

    if omega is not None:
        if omega > 0:
            # long range part
            coulG *= np.exp(-.25/omega**2 * absG2)
        elif omega < 0:
            # short range part
            coulG *= (1 - np.exp(-.25/omega**2 * absG2))
    elif cell.omega > 0:
        coulG *= np.exp(-.25/cell.omega**2 * absG2)
    elif cell.omega < 0:
        raise NotImplementedError

    return coulG

get_monkhorst_pack_size = stop_trace(pyscf_pbctools.get_monkhorst_pack_size)
cutoff_to_mesh = stop_trace(pyscf_pbctools.cutoff_to_mesh)

def madelung(cell, kpts, omega=None):
    Nk = get_monkhorst_pack_size(cell, kpts)
    ecell = cell.copy()
    ecell.coords = numpy.array([[0., 0., 0.],])
    ecell._atm = numpy.array([[1, cell._env.size, 0, 0, 0, 0]])
    ecell._env = numpy.append(cell._env, [0., 0., 0.])
    ecell.unit = 'B'
    ecell.a = a = np.einsum('xi,x->xi', cell.lattice_vectors(), Nk)

    if cell.omega == 0:
        return -2*ecell.ewald()
    else:
        precision = cell.precision
        Ecut = 10.
        Ecut = numpy.log(16*numpy.pi**2/(2*omega**2*(2*Ecut)**.5) / precision + 1.) * 2*omega**2
        Ecut = numpy.log(16*numpy.pi**2/(2*omega**2*(2*Ecut)**.5) / precision + 1.) * 2*omega**2
        mesh = cutoff_to_mesh(a, Ecut)
        Gv, Gvbase, weights = ecell.get_Gv_weights(mesh)
        wcoulG = get_coulG(ecell, Gv=Gv, omega=abs(omega), exxdiv=None) * weights
        SI = ecell.get_SI(mesh=mesh)
        ZSI = SI[0]
        e_lr = (2*abs(omega)/numpy.pi**0.5 -
                numpy.einsum('i,i,i->', ZSI.conj(), ZSI, wcoulG).real)
        if omega > 0:
            return e_lr
        else:
            e_fr = -2*ecell.ewald() # The full-range Coulomb
            return e_fr - e_lr
