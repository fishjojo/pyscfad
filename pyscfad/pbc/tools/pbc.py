from functools import wraps
import warnings
import copy
import numpy
import numpy as np
from pyscf import lib
from pyscf.pbc import tools as pyscf_pbctools
from pyscfad import numpy as jnp
from pyscfad import ops
from pyscfad.ops import stop_grad, stop_trace
from pyscfad.lib import logger

@wraps(pyscf_pbctools.fft)
def fft(f, mesh):
    if f.size == 0:
        return np.zeros_like(f)

    f3d = f.reshape(-1, *mesh)
    assert(f3d.shape[0] == 1 or f[0].size == f3d[0].size)
    g3d = jnp.fft.fftn(f3d, axes=(1,2,3))
    ngrids = np.prod(mesh)
    if f.ndim == 1 or (f.ndim == 3 and f.size == ngrids):
        return g3d.ravel()
    else:
        return g3d.reshape(-1, ngrids)

@wraps(pyscf_pbctools.ifft)
def ifft(g, mesh):
    if g.size == 0:
        return np.zeros_like(g)

    g3d = g.reshape(-1, *mesh)
    assert(g3d.shape[0] == 1 or g[0].size == g3d[0].size)
    f3d = jnp.fft.ifftn(g3d, axes=(1,2,3))
    ngrids = np.prod(mesh)
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

def cutoff_to_mesh(a, cutoff):
    """Batched version of :func:`pyscf.pbc.tools.cutoff_to_mesh`
    """
    a = numpy.asarray(a)
    cutoff = numpy.asarray(cutoff)

    b = 2 * numpy.pi * numpy.linalg.inv(a.T)
    B = numpy.dot(b, b.T)
    w, v = numpy.linalg.eigh(B)
    Gmax = numpy.einsum("xy,...y->...x", v, numpy.sqrt(2 * cutoff[...,None] / w))

    mesh = numpy.ceil(Gmax).astype(int) * 2 + 1
    return mesh

# modified from pyscf v2.6
@wraps(pyscf_pbctools.get_lattice_Ls)
def get_lattice_Ls(cell, nimgs=None, rcut=None, dimension=None, discard=True):
    if dimension is None:
        if cell.dimension < 2 or cell.low_dim_ft_type == 'inf_vacuum':
            dimension = cell.dimension
        else:
            dimension = 3
    if rcut is None:
        rcut = cell.rcut

    if dimension == 0 or rcut <= 0:
        return np.zeros((1, 3))

    a1 = cell.lattice_vectors()
    a = ops.to_numpy(a1)

    scaled_atom_coords = ops.to_numpy(cell.get_scaled_atom_coords())
    atom_boundary_max = scaled_atom_coords[:,:dimension].max(axis=0)
    atom_boundary_min = scaled_atom_coords[:,:dimension].min(axis=0)
    if (np.any(atom_boundary_max > 1) or np.any(atom_boundary_min < -1)):
        atom_boundary_max[atom_boundary_max > 1] = 1
        atom_boundary_min[atom_boundary_min <-1] = -1
    ovlp_penalty = atom_boundary_max - atom_boundary_min
    dR = ovlp_penalty.dot(a[:dimension])
    dR_basis = np.diag(dR)

    def find_boundary(a):
        aR = np.vstack([a, dR_basis])
        r = np.linalg.qr(aR.T)[1]
        ub = (rcut + abs(r[2,3:]).sum()) / abs(r[2,2])
        return ub

    xb = find_boundary(a[[1,2,0]])
    if dimension > 1:
        yb = find_boundary(a[[2,0,1]])
    else:
        yb = 0
    if dimension > 2:
        zb = find_boundary(a)
    else:
        zb = 0
    bounds = np.ceil([xb, yb, zb]).astype(int)
    Ts = lib.cartesian_prod((np.arange(-bounds[0], bounds[0]+1),
                             np.arange(-bounds[1], bounds[1]+1),
                             np.arange(-bounds[2], bounds[2]+1)))
    Ls = jnp.dot(Ts[:,:dimension], a1[:dimension])

    if discard:
        ovlp_penalty += 1e-200  # avoid /0
        Ts_scaled = (Ts[:,:dimension] + 1e-200) / ovlp_penalty
        ovlp_penalty_fac = 1. / abs(Ts_scaled).min(axis=1)
        Ls_mask = np.linalg.norm(stop_grad(Ls), axis=1) * (1-ovlp_penalty_fac) < rcut
        Ls = Ls[Ls_mask]
    return jnp.asarray(Ls)


@wraps(pyscf_pbctools.get_coulG)
def get_coulG(cell, k=np.zeros(3), exx=False, mf=None, mesh=None, Gv=None,
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

    equal2boundary = np.zeros(Gv.shape[0], dtype=bool)
    if wrap_around and abs(k).sum() > 1e-9:
        b = cell.reciprocal_vectors()
        box_edge = jnp.einsum('i,ij->ij', np.asarray(mesh)//2+0.5, b)
        assert (all(stop_trace(np.linalg.solve)(box_edge.T, k).round(9).astype(int)==0))
        reduced_coords = stop_trace(np.linalg.solve)(box_edge.T, kG.T).T.round(9)
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

    absG2 = jnp.einsum('gi,gi->g', kG, kG)
    G0_idx = jnp.where(absG2==0)[0]
    absG2 = jnp.where(absG2!=0, absG2, 0)

    if getattr(mf, 'kpts', None) is not None:
        kpts = mf.kpts
    else:
        kpts = k.reshape(1,3)
    Nk = len(kpts)

    if exxdiv == 'vcut_sph':  # PRB 77 193110
        Rc = (3*Nk*cell.vol/(4*np.pi))**(1./3)
        with np.errstate(divide='ignore',invalid='ignore'):
            coulG = 4*np.pi/absG2*(1.0 - jnp.cos(jnp.sqrt(absG2)*Rc))
        if len(G0_idx) > 0:
            coulG = ops.index_update(coulG, ops.index[G0_idx], 4*np.pi*0.5*Rc**2)

        if cell.dimension < 3:
            raise NotImplementedError
    elif exxdiv == 'vcut_ws':  # PRB 87, 165122
        raise NotImplementedError
    else:
        # Ewald probe charge method to get the leading term of the finite size
        # error in exchange integrals

        if cell.dimension != 2 or cell.low_dim_ft_type == 'inf_vacuum':
            with np.errstate(divide='ignore'):
                coulG = 4*np.pi/absG2
            if len(G0_idx) > 0:
                coulG = ops.index_update(coulG, ops.index[G0_idx], 0)

        elif cell.dimension == 2:
            # The following 2D analytical fourier transform is taken from:
            # R. Sundararaman and T. Arias PRB 87, 2013
            b = cell.reciprocal_vectors()
            Ld2 = np.pi/jnp.linalg.norm(b[2])
            Gz = kG[:,2]
            Gp = jnp.linalg.norm(kG[:,:2], axis=1)
            weights = 1. - jnp.cos(Gz*Ld2) * jnp.exp(-Gp*Ld2)
            with np.errstate(divide='ignore', invalid='ignore'):
                coulG = weights*4*np.pi/absG2
            if len(G0_idx) > 0:
                coulG = ops.index_update(coulG, ops.index[G0_idx], -2*np.pi*Ld2**2)

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
            coulG *= jnp.exp(-.25/omega**2 * absG2)
        elif omega < 0:
            # short range part
            coulG *= (1 - jnp.exp(-.25/omega**2 * absG2))
    elif cell.omega > 0:
        coulG *= jnp.exp(-.25/cell.omega**2 * absG2)
    elif cell.omega < 0:
        raise NotImplementedError

    return coulG

get_monkhorst_pack_size = stop_trace(pyscf_pbctools.get_monkhorst_pack_size)

def madelung(cell, kpts):
    Nk = get_monkhorst_pack_size(cell, kpts)
    ecell = copy.copy(cell)
    ecell._atm = np.array([[1, cell._env.size, 0, 0, 0, 0]])
    ecell._env = np.append(cell._env, [0., 0., 0.])
    ecell.unit = 'B'
    ecell.a = a = jnp.einsum('xi,x->xi', cell.lattice_vectors(), Nk)

    if cell.omega == 0:
        return -2*ecell.ewald()
    else:
        precision = cell.precision
        omega = cell.omega
        Ecut = 10.
        Ecut = np.log(16*np.pi**2/(2*omega**2*(2*Ecut)**.5) / precision + 1.) * 2*omega**2
        Ecut = np.log(16*np.pi**2/(2*omega**2*(2*Ecut)**.5) / precision + 1.) * 2*omega**2
        mesh = cutoff_to_mesh(a, Ecut)
        Gv, Gvbase, weights = ecell.get_Gv_weights(mesh)
        wcoulG = get_coulG(ecell, Gv=Gv) * weights
        SI = ecell.get_SI(mesh=mesh)
        ZSI = SI[0]
        return 2*omega/np.pi**0.5-jnp.einsum('i,i,i->', ZSI.conj(), ZSI, wcoulG).real
