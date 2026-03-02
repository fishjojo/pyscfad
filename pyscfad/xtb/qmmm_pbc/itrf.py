# Copyright 2025-2026 The PySCFAD Authors
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

import functools
import jax
from jax.lax import stop_gradient, scan
from typing import Tuple

from pyscf import lib
from pyscf.gto.mole import is_au

from pyscfad import numpy
from pyscfad.pbc.gto import cell
from pyscfad.xtb.xtb import mulliken_charge, util
from pyscfad.dft.rks import VXC

from pyscfad.scipy.special import erfc


@functools.partial(jax.custom_jvp, nondiff_argnums=(1, 2))
def lambertw(
    z: numpy.ndarray, tol: float = 1e-8, max_iter: int = 100
) -> numpy.ndarray:
    """Principal branch of the
    `Lambert W function <https://en.wikipedia.org/wiki/Lambert_W_function>`_.

    This implementation uses Halley's iteration and the global initialization
    proposed in :cite:`iacono:17`, Eq. 20 .

    Args:
       z: Array.
       tol: Tolerance threshold.
       max_iter: Maximum number of iterations.

    Returns:
      The Lambert W evaluated at ``z``.
    """  # noqa: D205

    def initial_iacono(x: numpy.ndarray) -> numpy.ndarray:
        y = numpy.sqrt(1.0 + numpy.e * x)
        num = 1.0 + 1.14956131 * y
        denom = 1.0 + 0.45495740 * numpy.log1p(y)
        return -1.0 + 2.036 * numpy.log(num / denom)

    def cond_fun(container):
        it, converged, _ = container
        return numpy.logical_and(numpy.any(~converged), it < max_iter)

    def halley_iteration(container):
        it, _, w = container

        # modified from `tensorflow_probability`
        f = w - z * numpy.exp(-w)
        delta = f / (w + 1.0 - 0.5 * (w + 2.0) * f / (w + 1.0))

        w_next = w - delta

        not_converged = numpy.abs(delta) <= tol * numpy.abs(w_next)
        return it + 1, not_converged, w_next

    w0 = initial_iacono(z)
    converged = numpy.zeros_like(w0, dtype=bool)

    _, _, w = jax.lax.while_loop(
        cond_fun=cond_fun, body_fun=halley_iteration, init_val=(
            0, converged, w0)
    )
    return w


@lambertw.defjvp
def _lambertw_jvp(
    tol: float, max_iter: int, primals: Tuple[numpy.ndarray, ...],
    tangents: Tuple[numpy.ndarray, ...]
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    z, = primals
    dz, = tangents
    w = lambertw(z, tol=tol, max_iter=max_iter)
    pz = numpy.where(z == 0.0, 1.0, w / ((1.0 + w) * z))
    return w, pz * dz


def _chunkize(data, chunk_size=1024):
    '''
    reshape data into (data.shape[0] / chunk_size, chunk_size, ...)
    pad zeros if not exact division
    '''
    n = data.shape[0]
    padsize = chunk_size - n % chunk_size
    pad_width = ((0, padsize), ) + ((0, 0), ) * (data.ndim - 1)
    return numpy.pad(data, pad_width, mode='constant').reshape(
        (-1, chunk_size, *data.shape[1:]))


def _structural_factor(Gv, coord_batches, charge_batches):
    '''
    charges dot cos(Gv dot coords)
    charges dot sin(Gv dot coords)
    '''
    @jax.checkpoint
    def body_fun(carry, input):
        coord_batch, charge_batch = input
        zcosGvR, zsinGvR = carry
        GvR = numpy.einsum('gx,ix->ig', Gv, coord_batch)
        cosGvR = numpy.cos(GvR)
        sinGvR = numpy.sin(GvR)
        zcosGvR += numpy.dot(charge_batch, cosGvR)
        zsinGvR += numpy.dot(charge_batch, sinGvR)
        return (zcosGvR, zsinGvR), None

    init_carry = (numpy.zeros(Gv.shape[0]), numpy.zeros(Gv.shape[0]))
    return scan(body_fun, init_carry, (coord_batches, charge_batches))[0]


def _bspline(u, order=6):
    '''
    B-splines evaluated on integer grids
    Distribute unity onto grids [x]-order//2+1, [x]-order//2+2, ..., [x]+order//2, depending on u = x - [x]

    Args:
        u: float
            real number in [0, 1]
        order: int
            B-spline order

    Returns:
        numpy.ndarray
            B-spline B(x, order) evaluated at u+order-1, u+order-2, ..., u
    '''
    if order == 4:
        u2 = u**2
        u3 = u2*u
        outputs = [
            1/6 - u/2 + u2/2 - u3/6,
            2/3 - u2 + u3/2,
            1/6 + u/2 + u2/2 - u3/2,
            u3/6
        ]
        return numpy.asarray(outputs)
    elif order == 6:
        u2 = u**2
        u3 = u2*u
        u4 = u3*u
        u5 = u4*u
        outputs = [
            1/120 - u/24 + u2/12 - u3/12 + u4/24 - u5/120,
            13/60 - (5*u)/12 + u2/6 + u3/6 - u4/6 + u5/24,
            11/20 - u2/2 + u4/4 - u5/12,
            13/60 + (5*u)/12 + u2/6 - u3/6 - u4/6 + u5/12,
            1/120 + u/24 + u2/12 + u3/12 + u4/24 - u5/24,
            u5/120
        ]
        return numpy.asarray(outputs)
    elif order == 10:
        outputs = [
            -2.7557319223985893e-6*(-1 + u)**9,
            (502 + 3*u*(-738 + u*(1416 + u*(-1512 + u *
             (924 + u*(-252 + u*(-56 + 3*u*(24 + (-8 + u)*u))))))))/362880.,
            (7304 - 3*u*(6069 + 2*u*(-2856 + u*(938 + u *
             (336 + u*(-357 + u*(56 + 3*u*(14 + (-7 + u)*u))))))))/181440.,
            (44117 + 21*u*(-2427 + 2*u*(66 + u*(434 + u *
             (-129 + u*(-69 + u*(34 + u*(6 + (-6 + u)*u))))))))/181440.,
            (78095 - 21*u**2*(2100 - 570*u**2 + 100*u**4 + 3*(-5 + u)*u**6))/181440.,
            (44117 + 21*u*(2427 + u*(132 + u*(-868 + u *
             (-258 + u*(138 + u*(68 + 3*u*(-4 + (-4 + u)*u))))))))/181440.,
            (7304 - 21*u*(-867 + 2*u*(-408 + u *
             (-134 + u*(48 + u*(51 + (-4 + u)*(-1 + u)*u*(2 + u)))))))/181440.,
            (251 + 3*u*(369 + 2*u*(354 + u*(378 + u *
             (231 + u*(63 + u*(-14 + 3*u*(-6 + (-2 + u)*u))))))))/181440.,
            (1 + 3*u*(3 + u*(12 + u*(28 + u*(42 + u*(42 + u*(28 + 3*u*(4 + u - u**2))))))))/362880.,
            u**9/362880.
        ]
        return numpy.asarray(outputs)
    else:
        raise NotImplementedError


def _bspline_deriv(u, order=6):
    '''
    _bspline first derivative
    '''
    if order == 4:
        u2 = u**2
        outputs = [
            -(1/2) + u - u2/2,
            -2*u + (3*u2)/2,
            1/2 + u - (3*u2)/2,
            u2/2
        ]
        return numpy.asarray(outputs)
    elif order == 6:
        u2 = u**2
        u3 = u2*u
        u4 = u3*u
        outputs = [
            -(1/24) + u/6 - u2/4 + u3/6 - u4/24,
            -(5/12) + u/3 + u2/2 - (2*u3)/3 + (5*u4)/24,
            -u + u3 - (5*u4)/12,
            5/12 + u/3 - u2/2 - (2*u3)/3 + (5*u4)/12,
            1/24 + u/6 + u2/4 + u3/6 - (5*u4)/24,
            u4/24
        ]
        return numpy.asarray(outputs)
    elif order == 10:
        outputs = [
            -(1-u)**8/40320.,
            -0.006101190476190476 + u*(0.023412698412698413 + u*(-0.0375 + u*(0.030555555555555555 + u*(-0.010416666666666666 +
                                       u*(-0.002777777777777778 + u*(0.004166666666666667 + (-0.0015873015873015873 + u/4480.)*u)))))),
            -0.10034722222222223 + u*(0.18888888888888888 + u*(-0.09305555555555556 + u*(-0.044444444444444446 + u*(
                0.059027777777777776 + u*(-0.011111111111111112 + u*(-0.009722222222222222 + (0.005555555555555556 - u/1120.)*u)))))),
            -0.2809027777777778 + u*(0.030555555555555555 + u*(0.3013888888888889 + u*(-0.11944444444444445 + u *
                                     (-0.0798611111111111 + u*(0.04722222222222222 + u*(0.009722222222222222 + (-0.011111111111111112 + u/480.)*u)))))),
            u*(-0.4861111111111111 + u**2*(0.2638888888888889 + u**2 *
               (-0.06944444444444445 + (0.013888888888888888 - u/320.)*u**2))),
            0.2809027777777778 + u*(0.030555555555555555 + u*(-0.3013888888888889 + u*(-0.11944444444444445 + u*(
                0.0798611111111111 + u*(0.04722222222222222 + u*(-0.009722222222222222 + (-0.011111111111111112 + u/320.)*u)))))),
            0.10034722222222223 + u*(0.18888888888888888 + u*(0.09305555555555556 + u*(-0.044444444444444446 + u *
                                     (-0.059027777777777776 + u*(-0.011111111111111112 + u*(0.009722222222222222 + (0.005555555555555556 - u/480.)*u)))))),
            0.006101190476190476 + u*(0.023412698412698413 + u*(0.0375 + u*(0.030555555555555555 + u*(
                0.010416666666666666 + u*(-0.002777777777777778 + u*(-0.004166666666666667 + (-0.0015873015873015873 + u/1120.)*u)))))),
            0.0000248015873015873 + u*(0.0001984126984126984 + u*(0.0006944444444444445 + u*(0.001388888888888889 + u*(
                0.001736111111111111 + u*(0.001388888888888889 + u*(0.0006944444444444445 + (0.0001984126984126984 - u/4480.)*u)))))),
            u**8/40320.
        ]
        return numpy.asarray(outputs)
    else:
        raise NotImplementedError


def _bspline_deriv2(u, order=6):
    '''
    _bspline second derivative
    '''
    if order == 4:
        outputs = [
            1 - u,
            -2 + 3 * u,
            1 - 3 * u,
            u
        ]
        return numpy.asarray(outputs)
    elif order == 6:
        u2 = u**2
        u3 = u2*u
        outputs = [
            1/6 - u/2 + u2/2 - u3/6,
            1/3 + u - 2*u2 + (5*u3)/6,
            -1 + 3*u2 - (5*u3)/3,
            1/3 - u - 2*u2 + (5*u3)/3,
            1/6 + u/2 + u2/2 - (5*u3)/6,
            u3/6
        ]
        return numpy.asarray(outputs)
    elif order == 10:
        outputs = [
            (1-u)**7/5040.,
            0.023412698412698413 + u*(-0.075 + u*(0.09166666666666666 + u*(-0.041666666666666664 +
                                      u*(-0.013888888888888888 + u*(0.025 + (-0.011111111111111112 + u/560.)*u))))),
            0.18888888888888888 + u*(-0.18611111111111112 + u*(-0.13333333333333333 + u*(0.2361111111111111 +
                                     u*(-0.05555555555555555 + u*(-0.058333333333333334 + (0.03888888888888889 - u/140.)*u))))),
            0.030555555555555555 + u*(0.6027777777777777 + u*(-0.35833333333333334 + u*(-0.3194444444444444 + u*(
                0.2361111111111111 + u*(0.058333333333333334 + (-0.07777777777777778 + u/60.)*u))))),
            -0.4861111111111111 + u**2 *
            (0.7916666666666666 + u**2 *
             (-0.3472222222222222 + (0.09722222222222222 - u/40.)*u**2)),
            0.030555555555555555 + u*(-0.6027777777777777 + u*(-0.35833333333333334 + u*(0.3194444444444444 + u*(
                0.2361111111111111 + u*(-0.058333333333333334 + (-0.07777777777777778 + u/40.)*u))))),
            0.18888888888888888 + u*(0.18611111111111112 + u*(-0.13333333333333333 + u*(-0.2361111111111111 +
                                     u*(-0.05555555555555555 + u*(0.058333333333333334 + (0.03888888888888889 - u/60.)*u))))),
            0.023412698412698413 + u*(0.075 + u*(0.09166666666666666 + u*(0.041666666666666664 +
                                      u*(-0.013888888888888888 + u*(-0.025 + (-0.011111111111111112 + u/140.)*u))))),
            0.0001984126984126984 + u*(0.001388888888888889 + u*(0.004166666666666667 + u*(0.006944444444444444 + u*(
                0.006944444444444444 + u*(0.004166666666666667 + (0.001388888888888889 - u/560.)*u))))),
            u**7/5040.
        ]
        return numpy.asarray(outputs)
    else:
        raise NotImplementedError


def _get_pme_mesh_indices(u, mesh, order=6):
    '''
    Compute grid indices and weights for PME

    Args:
        u: (N, 3)
            fractional coordinate(s) measured in unit cell
        mesh:  (3, )
    '''

    scaled = u * numpy.asarray(mesh)  # (N, 3)
    idx_floor = numpy.floor(scaled).astype(int)
    du = scaled - idx_floor

    # weights: 3 x (N, order)
    wx = numpy.stack(_bspline(du[:, 0], order), axis=-1)
    wy = numpy.stack(_bspline(du[:, 1], order), axis=-1)
    wz = numpy.stack(_bspline(du[:, 2], order), axis=-1)

    # indices: 3 x (N, order)
    # offset for order 4: -1, 0, 1, 2
    # offset for order 6: -2, -1, 0, 1, 2, 3
    offsets = numpy.arange(-order//2+1, order//2+1)
    idx_x = (idx_floor[:, 0:1] + offsets) % mesh[0]
    idx_y = (idx_floor[:, 1:2] + offsets) % mesh[1]
    idx_z = (idx_floor[:, 2:3] + offsets) % mesh[2]

    return (idx_x, idx_y, idx_z), (wx, wy, wz)


def _get_pme_mesh_indices_deriv(u, mesh, order=6):
    scaled = u * numpy.asarray(mesh)
    idx_floor = numpy.floor(scaled).astype(int)
    du = scaled - idx_floor

    dwx = numpy.stack(_bspline_deriv(du[:, 0], order), axis=-1)
    dwy = numpy.stack(_bspline_deriv(du[:, 1], order), axis=-1)
    dwz = numpy.stack(_bspline_deriv(du[:, 2], order), axis=-1)

    offsets = numpy.arange(-1, order-1)
    idx_x = (idx_floor[:, 0:1] + offsets) % mesh[0]
    idx_y = (idx_floor[:, 1:2] + offsets) % mesh[1]
    idx_z = (idx_floor[:, 2:3] + offsets) % mesh[2]
    return (idx_x, idx_y, idx_z), (dwx, dwy, dwz)


def _get_pme_mesh_indices_hess(u, mesh, order=6):
    scaled = u * numpy.asarray(mesh)
    idx_floor = numpy.floor(scaled).astype(int)
    du = scaled - idx_floor

    d2wx = numpy.stack(_bspline_deriv2(du[:, 0], order), axis=-1)
    d2wy = numpy.stack(_bspline_deriv2(du[:, 1], order), axis=-1)
    d2wz = numpy.stack(_bspline_deriv2(du[:, 2], order), axis=-1)

    offsets = numpy.arange(-1, order-1)
    idx_x = (idx_floor[:, 0:1] + offsets) % mesh[0]
    idx_y = (idx_floor[:, 1:2] + offsets) % mesh[1]
    idx_z = (idx_floor[:, 2:3] + offsets) % mesh[2]
    return (idx_x, idx_y, idx_z), (d2wx, d2wy, d2wz)


def _get_pme_correction(mesh, order=6):
    def get_b(ng):
        m = numpy.arange(ng)
        vals = _bspline(1., order=order)[1:]
        args = 2 * numpy.pi * 1j * \
            m[:, None] * numpy.arange(order-1)[None, :] / ng  # (Ng, order-1)
        denom = numpy.sum(vals * numpy.exp(args), axis=-1)
        numer = numpy.exp(2 * numpy.pi * 1j * (order - 1) * m / ng)
        b = numer / denom
        return numpy.abs(b)**2

    bx = get_b(mesh[0])
    by = get_b(mesh[1])
    bz = get_b(mesh[2])
    return numpy.einsum('x,y,z->xyz', bx, by, bz)


def add_mm_charges(xtb_method, mm_coords, a, mm_charges, mm_radii,
                   mm_ew_eta=None, mm_ew_rcut=None, mm_ew_mesh=None, mm_pme_order=10,  max_mm_nbr=None,
                   qm_ew_mesh=None, pbcqm=True,
                   ew_precision=1e-6, unit=None):
    if unit is None:
        unit = xtb_method.mol.unit
    if not is_au(unit):
        mm_coords = mm_coords / lib.param.BOHR
        a = a / lib.param.BOHR
        mm_radii = mm_radii / lib.param.BOHR
        rcut = rcut / lib.param.BOHR
    xtbqmmm = QMMM(xtb_method,
                   mm_coords=mm_coords, a=a, mm_charges=mm_charges, mm_radii=mm_radii,
                   ew_precision=ew_precision,
                   max_mm_nbr=max_mm_nbr,
                   mm_ew_eta=mm_ew_eta, mm_ew_rcut=mm_ew_rcut, mm_ew_mesh=mm_ew_mesh, qm_ew_mesh=qm_ew_mesh,
                   pbcqm=pbcqm,
                   mm_pbe_order=mm_pme_order,
                   )
    return lib.set_class(xtbqmmm, (QMMM, xtb_method.__class__))


class QMMM:
    def __init__(self, method, mm_coords, a, mm_charges, mm_radii,
                 mm_ew_eta=None, mm_ew_rcut=None, mm_ew_mesh=None, mm_pbe_order=10, max_mm_nbr=None,
                 qm_ew_mesh=None, pbcqm=True,
                 ew_precision=1e-6):
        '''
        pbcqm: whether compute electronic qm-qm PBC interactions
        '''
        self.__dict__.update(method.__dict__)
        self.pbcqm = pbcqm
        self.s1 = None
        self.s1r = None
        self.s1rr = None
        self.mm_ewald_pot = None
        self.qm_ewald_hess = None

        self.a = a
        self.mm_coords = mm_coords
        self.mm_charges = mm_charges
        self.mm_radii = mm_radii

        self.ew_precision = ew_precision
        self.max_mm_nbr = max_mm_nbr
        self.mm_ew_eta = mm_ew_eta
        self.mm_ew_rcut = mm_ew_rcut
        self.mm_ew_mesh = mm_ew_mesh
        self.qm_ew_mesh = qm_ew_mesh

        self.mm_pbe_order = mm_pbe_order
        self.dimension = 3

        # determine rcut as the mininum distance from unit cell boundaries
        a1, a2, a3 = self.a
        area_1 = numpy.linalg.norm(numpy.cross(a2, a3))
        area_2 = numpy.linalg.norm(numpy.cross(a1, a3))
        area_3 = numpy.linalg.norm(numpy.cross(a1, a2))
        widths = self.vol / numpy.asarray([area_1, area_2, area_3])
        coords = self.mol.atom_coords()
        coords -= numpy.mean(coords, axis=0)
        coords += numpy.dot(numpy.array([0.5] * 3),  self.a)
        reduce_coords = numpy.linalg.solve(self.a.T, coords.T).T
        dist0 = reduce_coords * widths[None]  # n x 3, 3 -> n x 3
        dist1 = (1.0 - reduce_coords) * widths[None]
        max_ew_rcut = numpy.min(numpy.hstack([dist0, dist1]))

        # Direct Ewald parameters for QM and QM
        Q = numpy.sum(numpy.abs(self.mol.atom_charges()))**2
        self.qm_ew_eta, qm_ew_mesh = self.get_ewald_params(
            Q, ew_rcut=max_ew_rcut)
        if self.qm_ew_mesh is None:
            self.qm_ew_mesh = qm_ew_mesh

        # PME parameters for QM and MM
        if self.mm_ew_rcut is None:
            self.mm_ew_rcut = max_ew_rcut
        else:
            # TODO raise warn if ew_rcut larger than max_ew_rcut
            pass
        if self.max_mm_nbr is None:
            self.max_mm_nbr = max(256, int(1024 * (self.mm_ew_rcut / 18.)**3))
        Q = numpy.sum(numpy.abs(self.mm_charges)) * \
            numpy.sum(numpy.abs(self.mol.atom_charges()))
        mm_ew_eta, mm_ew_mesh = self.get_ewald_params(
            Q, ew_rcut=self.mm_ew_rcut)
        if self.mm_ew_eta is None:
            self.mm_ew_eta = mm_ew_eta
        if self.mm_ew_mesh is None:
            self.mm_ew_mesh = mm_ew_mesh

        # TODO print out the Ewald error estimate for both QM-QM and QM-MM

    @property
    def vol(self):
        return abs(numpy.linalg.det(self.a))

    get_Gv_weights = cell.Cell.get_Gv_weights
    reciprocal_vectors = cell.Cell.reciprocal_vectors

    def lattice_vectors(self):
        return self.a

    def get_ewald_params(self, Q, ew_rcut=None, precision=None):
        if precision is None:
            precision = self.ew_precision
        if ew_rcut is None:
            ew_rcut = self.ew_rcut

        e = precision
        eta = stop_gradient(
            1 / ew_rcut * numpy.sqrt(
                1.5 *
                lambertw(
                    2/3 * (4/e*Q/ew_rcut/self.vol)**(2/3) *
                    ew_rcut**2
                ).real
            )
        )
        L = self.vol**(1/3)
        kmax = 1.73205081*eta/2/numpy.pi * numpy.sqrt(
            lambertw(4*Q**(2/3)/3/numpy.pi**(2/3)/L**2/eta**(2/3) / e**(4/3)).real)
        mesh = stop_gradient(numpy.asarray(numpy.ceil(
            numpy.diag(self.a) * kmax) * 2 + 1, dtype=int))

        return eta, mesh

    def get_mm_ewald_pot(self, param=None):
        if param is None:
            param = self.param
        ew_eta, mesh = self.mm_ew_eta, self.mm_ew_mesh

        coords1 = self.mol.atom_coords()
        coords2 = self.mm_coords

        if len(coords2) > self.max_mm_nbr:
            r2 = -numpy.sum((coords1[:, None, :] - coords2[None])**2, axis=-1)
            _, neighbors = jax.lax.stop_gradient(
                jax.lax.top_k(r2, k=self.max_mm_nbr, axis=-1))
        else:
            neighbors = numpy.array(
                [numpy.arange(len(coords2))] * len(coords1))

        coords2 = coords2[neighbors]  # shape = Nqm, max_mm_nbr, 3
        mm_charges = self.mm_charges[neighbors]
        mm_radii = self.mm_radii[neighbors]

        atom_to_bas = util.atom_to_bas_indices(self.mol)

        (ewovrl0, ewovrl1, ewovrl2) = (
            numpy.zeros_like(param.gam),
            numpy.zeros((len(coords1), 3)),
            numpy.zeros((len(coords1), 3, 3))
        )

        R = coords1[:, None, :] - coords2
        r2 = numpy.sum(R * R, axis=-1)
        r = numpy.sqrt(numpy.where(r2 < 1e-20, numpy.inf, r2))

        # TODO raise warning if max(r) < self.ew_rcut

        # difference between MM gaussain charges and MM point charges
        # TODO since Ewald rcut and the following expnts are fixed,
        # need to check if max_mm_nbr can give desired ewald precision
        expnts = 2. / (1 / (param.gam*param.lgam)
                       [:, None] + mm_radii[atom_to_bas])
        Tij = erfc(expnts * r[atom_to_bas]) / r[atom_to_bas]
        ewovrl0 -= numpy.einsum('ij,ij->i', Tij, mm_charges[atom_to_bas])
        if param.dipgam is not None:
            expnts = 2. / (1 / param.dipgam[:, None] + mm_radii)
            ekR = numpy.exp(-expnts**2 * r**2)
            Tij = erfc(expnts * r) / r
            invr3 = (Tij + 2/numpy.sqrt(numpy.pi) * expnts * ekR) / r**2
            Tija = -numpy.einsum('ijx,ij->ijx', R, invr3)
            ewovrl1 -= numpy.einsum('ij,ija->ia', mm_charges, Tija)
        if param.quadgam is not None:
            expnts = 2. / (1 / param.quadgam[:, None] + mm_radii)
            ekR = numpy.exp(-expnts**2 * r**2)
            Tij = erfc(expnts * r) / r
            invr3 = (Tij + 2/numpy.sqrt(numpy.pi) * expnts * ekR) / r**2
            Tija = -numpy.einsum('ijx,ij->ijx', R, invr3)
            Tijab = 3 * numpy.einsum('ija,ijb,ij->ijab', R, R, 1/r**2)
            Tijab -= numpy.einsum('ij,ab->ijab',
                                  numpy.ones_like(r), numpy.eye(3))
            invr5 = invr3 + 4/3/numpy.sqrt(numpy.pi) * expnts**3 * ekR
            Tijab = numpy.einsum('ijab,ij->ijab', Tijab, invr5)
            Tijab += numpy.einsum('ij,ij,ab->ijab', expnts **
                                  3, 4/3/numpy.sqrt(numpy.pi)*ekR, numpy.eye(3))
            ewovrl2 -= numpy.einsum('ij,ijab->iab', mm_charges, Tijab) / 3

        # ewald real-space sum; treat MM as point charges
        ekR = numpy.exp(-ew_eta**2 * r**2)
        # Tij = \hat{1/r} = f0 / r = erfc(r) / r
        Tij = erfc(ew_eta * r) / r
        ewovrl0 += numpy.einsum('ij,ij->i', Tij, mm_charges)[atom_to_bas]
        if param.dipgam is not None:
            # Tija = -Rija \hat{1/r^3} = -Rija / r^2 ( \hat{1/r} + 2 eta/sqrt(pi) exp(-eta^2 r^2) )
            invr3 = (Tij + 2*ew_eta/numpy.sqrt(numpy.pi) * ekR) / r**2
            Tija = -numpy.einsum('ijx,ij->ijx', R, invr3)
            ewovrl1 += numpy.einsum('ijx,ij->ix', Tija, mm_charges)
        if param.quadgam is not None:
            # Tijab = (3 RijaRijb - Rij^2 delta_ab) \hat{1/r^5}
            Tijab = 3 * numpy.einsum('ija,ijb,ij->ijab', R, R, 1/r**2)
            Tijab -= numpy.einsum('ij,ab->ijab',
                                  numpy.ones_like(r), numpy.eye(3))
            invr5 = invr3 + 4/3*ew_eta**3 / \
                numpy.sqrt(numpy.pi) * ekR  # NOTE this is invr5 * r**2
            Tijab = numpy.einsum('ijab,ij->ijab', Tijab, invr5)
            # NOTE the below is present in Eq 8 but missing in Eq 12
            Tijab += 4/3*ew_eta**3 / \
                numpy.sqrt(numpy.pi) * \
                numpy.einsum('ij,ab->ijab', ekR, numpy.eye(3))
            ewovrl2 += numpy.einsum('ijxy,ij->ixy', Tijab, mm_charges/3)

        # g-space sum (using FFT)
        ewg0, ewg1, ewg2 = self.get_mm_ewald_g_fft(
            param, mesh, ew_eta, coords1, self.mm_coords, self.mm_charges)

        return ewovrl0+ewg0[atom_to_bas], ewovrl1+ewg1, ewovrl2+ewg2

    def get_mm_ewald_g_fft(self, param, mesh, ew_eta, qm_coords, mm_coords, mm_charges):
        coords1 = qm_coords
        coords2 = mm_coords
        charges2 = mm_charges

        # 1. Spread MM charges to grid
        inv_a = numpy.linalg.inv(self.a)
        frac_coords2 = numpy.dot(coords2, inv_a)
        frac_coords2 = frac_coords2 % 1.0

        (idx_x, idx_y, idx_z), (wx, wy, wz) = _get_pme_mesh_indices(
            frac_coords2, mesh, order=self.mm_pbe_order)

        # Tensor product of weights
        # w: (Natom, order, order, order)
        w = numpy.einsum('ni,nj,nk->nijk', wx, wy, wz)

        # Scatter add to grid
        # grid_idx: 3x (Natom, order, order, order)
        grid_idx_x = idx_x[:, :, None, None]
        grid_idx_y = idx_y[:, None, :, None]
        grid_idx_z = idx_z[:, None, None, :]

        # Flatten for scatter
        q_spread = (charges2[:, None, None, None] * w).ravel()
        idx_flat_x = numpy.broadcast_to(grid_idx_x, w.shape).ravel()
        idx_flat_y = numpy.broadcast_to(grid_idx_y, w.shape).ravel()
        idx_flat_z = numpy.broadcast_to(grid_idx_z, w.shape).ravel()

        # Charge on grids
        grid = numpy.zeros(mesh, dtype=charges2.dtype)
        grid = grid.at[idx_flat_x, idx_flat_y, idx_flat_z].add(q_spread)

        # FFT to get structural factor
        Qk = numpy.fft.fftn(grid)

        # 3. Multiply by kernel
        Gv, _, _ = self.get_Gv_weights(mesh)
        absG2 = numpy.sum(Gv**2, axis=-1)
        # Avoid division by zero at G=0
        absG2 = numpy.where(absG2 == 0, 1e200, absG2)

        kernel = 4*numpy.pi / absG2 * mesh[0]*mesh[1]*mesh[2] / self.vol
        kernel = kernel * numpy.exp(-absG2/(4*ew_eta**2))
        kernel = kernel.reshape(*mesh)

        B = _get_pme_correction(mesh, order=self.mm_pbe_order)
        kernel *= B
        # Exclude G=0
        kernel = kernel.at[0, 0, 0].set(0)

        # 4. IFFT to get MM potential on grid
        pot_grid = numpy.fft.ifftn(Qk * kernel).real

        # 5. Interpolate at QM atoms
        frac_coords1 = numpy.dot(coords1, inv_a)
        (idx_x, idx_y, idx_z), (wx, wy, wz) = _get_pme_mesh_indices(
            frac_coords1, mesh, order=self.mm_pbe_order)

        # Gather potential
        # val: (N, order, order, order)
        # We need to gather from pot_grid using advanced indexing
        # idx arrays are (N, order). Broadcast to (N, order, order, order)
        g_idx_x = idx_x[:, :, None, None]
        g_idx_y = idx_y[:, None, :, None]
        g_idx_z = idx_z[:, None, None, :]

        val = pot_grid[g_idx_x, g_idx_y, g_idx_z]

        # Interpolate potential
        w = numpy.einsum('ni,nj,nk->nijk', wx, wy, wz)
        ewg0 = numpy.sum(val * w, axis=(1, 2, 3))

        # Interpolate dipole and quadrupole potential
        if param.dipgam is not None or param.quadgam is not None:
            (_, _, _), (dwx, dwy, dwz) = _get_pme_mesh_indices_deriv(
                frac_coords1, mesh, order=self.mm_pbe_order)

            # Transform to Cartesian gradient: dV/dr = dV/du * du/dr
            # u = r @ inv_a * mesh
            # du/dr = inv_a * mesh
            # dV/dr_alpha = sum_beta dV/du_beta * (inv_a)_beta,alpha * mesh_beta
            # inv_a is (3, 3) -> (beta, alpha) if r is row vector?
            # r_frac = r_cart @ inv_a. r_frac_i = r_cart_j * inv_a_ji.
            # u_i = r_frac_i * mesh_i.
            # du_i / dr_cart_j = inv_a_ji * mesh_i.

            # dV_dr: (N, 3)
            # dV_dr_j = sum_i dV_du_i * inv_a_ji * mesh_i
            metric = inv_a * numpy.asarray(mesh)[:, None]  # (i, j)

        if param.dipgam is not None:
            # Gradient w.r.t fractional coordinates u
            # dV/du_x = sum V_grid * dwx * wy * wz
            dV_dux = numpy.sum(
                val * numpy.einsum('ni,nj,nk->nijk', dwx, wy, wz), axis=(1, 2, 3))
            dV_duy = numpy.sum(
                val * numpy.einsum('ni,nj,nk->nijk', wx, dwy, wz), axis=(1, 2, 3))
            dV_duz = numpy.sum(
                val * numpy.einsum('ni,nj,nk->nijk', wx, wy, dwz), axis=(1, 2, 3))

            dV_du = numpy.stack([dV_dux, dV_duy, dV_duz], axis=-1)

            ewg1 = numpy.einsum('ni,ij->nj', dV_du, metric)
        else:
            ewg1 = 0

        if param.quadgam is not None:
            # Hessian
            # d2V/du2
            (_, _, _), (d2wx, d2wy, d2wz) = _get_pme_mesh_indices_hess(
                frac_coords1, mesh, order=self.mm_pbe_order)

            # xx
            d2V_duxx = numpy.sum(
                val * numpy.einsum('ni,nj,nk->nijk', d2wx, wy, wz), axis=(1, 2, 3))
            # yy
            d2V_duyy = numpy.sum(
                val * numpy.einsum('ni,nj,nk->nijk', wx, d2wy, wz), axis=(1, 2, 3))
            # zz
            d2V_duzz = numpy.sum(
                val * numpy.einsum('ni,nj,nk->nijk', wx, wy, d2wz), axis=(1, 2, 3))
            # xy
            d2V_duxy = numpy.sum(
                val * numpy.einsum('ni,nj,nk->nijk', dwx, dwy, wz), axis=(1, 2, 3))
            # xz
            d2V_duxz = numpy.sum(
                val * numpy.einsum('ni,nj,nk->nijk', dwx, wy, dwz), axis=(1, 2, 3))
            # yz
            d2V_duyz = numpy.sum(
                val * numpy.einsum('ni,nj,nk->nijk', wx, dwy, dwz), axis=(1, 2, 3))

            d2V_du2 = numpy.zeros((len(coords1), 3, 3))
            d2V_du2 = d2V_du2.at[:, 0, 0].set(d2V_duxx)
            d2V_du2 = d2V_du2.at[:, 1, 1].set(d2V_duyy)
            d2V_du2 = d2V_du2.at[:, 2, 2].set(d2V_duzz)
            d2V_du2 = d2V_du2.at[:, 0, 1].set(
                d2V_duxy).at[:, 1, 0].set(d2V_duxy)
            d2V_du2 = d2V_du2.at[:, 0, 2].set(
                d2V_duxz).at[:, 2, 0].set(d2V_duxz)
            d2V_du2 = d2V_du2.at[:, 1, 2].set(
                d2V_duyz).at[:, 2, 1].set(d2V_duyz)

            ewg2 = numpy.einsum('ji,njk,kl->nil', metric, d2V_du2, metric) / 3
        else:
            ewg2 = 0

        return ewg0, ewg1, ewg2

    def get_mm_ewald_g_direct(self, param, mesh, ew_eta, qm_coords, mm_coords, mm_charges):
        coords1 = qm_coords

        # g-space sum (using g grid)
        Gv, Gvbase, weights = self.get_Gv_weights(mesh)
        absG2 = numpy.einsum('gx,gx->g', Gv, Gv)
        absG2 = numpy.where(absG2 == 0, 1e200, absG2)

        coulG = 4*numpy.pi / absG2
        coulG *= weights
        # NOTE Gpref is actually Gpref*2
        Gpref = numpy.exp(-absG2/(4*ew_eta**2)) * coulG

        coord2_batches = _chunkize(mm_coords)
        mm_charge_batches = _chunkize(mm_charges)

        zcosGvR2, zsinGvR2 = _structural_factor(
            Gv, coord2_batches, mm_charge_batches)

        GvR1 = numpy.einsum('gx,ix->ig', Gv, coords1)
        cosGvR1 = numpy.cos(GvR1)
        sinGvR1 = numpy.sin(GvR1)
        # qm pc - mm pc
        ewg0 = numpy.einsum('ig,g,g->i', cosGvR1, zcosGvR2, Gpref)
        ewg0 += numpy.einsum('ig,g,g->i', sinGvR1, zsinGvR2, Gpref)
        ewg0 = ewg0
        # qm dip - mm pc
        if param.dipgam is not None:
            p = [(2, 3), (0, 2), (0, 1)]
            ewg1 = numpy.einsum('gx,ig,g,g->ix', Gv, cosGvR1,
                                zsinGvR2, Gpref, optimize=p)
            ewg1 -= numpy.einsum('gx,ig,g,g->ix', Gv,
                                 sinGvR1, zcosGvR2, Gpref, optimize=p)
        else:
            ewg1 = 0.
        # qm quad - mm pc
        if param.quadgam is not None:
            p = [(3, 4), (0, 3), (0, 2), (0, 1)]
            ewg2 = -numpy.einsum('gx,gy,ig,g,g->ixy', Gv,
                                 Gv, cosGvR1, zcosGvR2, Gpref, optimize=p)
            ewg2 += -numpy.einsum('gx,gy,ig,g,g->ixy', Gv,
                                  Gv, sinGvR1, zsinGvR2, Gpref, optimize=p)
            ewg2 /= 3
        else:
            ewg2 = 0.
        return ewg0, ewg1, ewg2

    def get_qm_ewald_hess(self):
        ew_eta, mesh = self.qm_ew_eta, self.qm_ew_mesh

        coords1 = self.mol.atom_coords()

        ewself00 = numpy.zeros((len(coords1), len(coords1)))
        ewself01 = numpy.zeros((len(coords1), len(coords1), 3))
        ewself11 = numpy.zeros((len(coords1), len(coords1), 3, 3))
        ewself02 = numpy.zeros((len(coords1), len(coords1), 3, 3))

        R = coords1[:, None] - coords1[None]
        r2 = numpy.sum(R * R, axis=-1)
        r2 = numpy.where(r2 < 1e-20, numpy.inf, r2)
        r = numpy.sqrt(r2)

        # ewald real-space sum; assumed rcut < image distances
        ekR = numpy.exp(-ew_eta**2 * r2)
        # Tij = \hat{1/r} = f0 / r = erfc(r) / r
        Tij = erfc(ew_eta * r) / r
        # Tija = -Rija \hat{1/r^3} = -Rija / r^2 ( \hat{1/r} + 2 eta/sqrt(pi) exp(-eta^2 r^2) )
        invr3 = (Tij + 2*ew_eta/numpy.sqrt(numpy.pi) * ekR) / r2
        Tija = -numpy.einsum('ijx,ij->ijx', R, invr3)
        # Tijab = (3 RijaRijb - Rij^2 delta_ab) \hat{1/r^5}
        Tijab = 3 * numpy.einsum('ija,ijb,ij->ijab', R, R, 1/r2)
        Tijab -= numpy.einsum('ij,ab->ijab', numpy.ones_like(r), numpy.eye(3))
        invr5 = invr3 + 4/3*ew_eta**3 / \
            numpy.sqrt(numpy.pi) * ekR  # NOTE this is invr5 * r**2
        Tijab = numpy.einsum('ijab,ij->ijab', Tijab, invr5)
        # NOTE the below is present in Eq 8 but missing in Eq 12
        Tijab += 4/3*ew_eta**3 / \
            numpy.sqrt(numpy.pi)*numpy.einsum('ij,ab->ijab', ekR, numpy.eye(3))
        ewself00 += Tij
        ewself01 -= Tija
        ewself11 -= Tijab
        ewself02 += Tijab / 3

        # unit cell Coloumb, to be subtracted out
        Tij = 1 / r
        Tija = -numpy.einsum('ijx,ij->ijx', R, Tij**3)
        Tijab = 3 * numpy.einsum('ija,ijb->ijab', R, R)
        Tijab = numpy.einsum('ijab,ij->ijab', Tijab, Tij**5)
        Tijab -= numpy.einsum('ij,ab->ijab', Tij**3, numpy.eye(3))
        ewself00 -= Tij
        ewself01 += Tija
        ewself11 += Tijab
        ewself02 -= Tijab / 3

        # spurious interaction with Ewald compensenting gaussian charge
        eye = numpy.eye(len(coords1))
        # -d^2 Eself / dQi dQj
        ewself00 += -eye * 2 * ew_eta / \
            numpy.sqrt(numpy.pi)  # to include the lim
        # -d^2 Eself / dDia dDjb
        ewself11 += -numpy.einsum('ij,ab->ijab', eye, numpy.eye(3)) \
            * 4 * ew_eta**3 / 3 / numpy.sqrt(numpy.pi)

        # g-space sum (using g grid)
        Gv, Gvbase, weights = self.get_Gv_weights(mesh)
        absG2 = numpy.einsum('gx,gx->g', Gv, Gv)
        absG2 = numpy.where(absG2 == 0, 1e200, absG2)

        coulG = 4*numpy.pi / absG2
        coulG *= weights
        # NOTE Gpref is actually Gpref*2
        Gpref = numpy.exp(-absG2/(4*ew_eta**2)) * coulG

        GvR = numpy.einsum('gx,ix->ig', Gv, coords1)
        cosGvR = numpy.cos(GvR)
        sinGvR = numpy.sin(GvR)

        # qm pc - qm pc
        ewg00 = numpy.einsum('ig,jg,g->ij', cosGvR, cosGvR, Gpref)
        ewg00 += numpy.einsum('ig,jg,g->ij', sinGvR, sinGvR, Gpref)
        # qm pc - qm dip
        ewg01 = numpy.einsum('gx,ig,jg,g->ijx', Gv, sinGvR, cosGvR, Gpref)
        ewg01 -= numpy.einsum('gx,ig,jg,g->ijx', Gv, cosGvR, sinGvR, Gpref)
        # qm dip - qm dip
        ewg11 = numpy.einsum('gx,gy,ig,jg,g->ijxy', Gv,
                             Gv, cosGvR, cosGvR, Gpref)
        ewg11 += numpy.einsum('gx,gy,ig,jg,g->ijxy', Gv,
                              Gv, sinGvR, sinGvR, Gpref)
        # qm pc - qm quad
        ewg02 = -numpy.einsum('gx,gy,ig,jg,g->ijxy', Gv,
                              Gv, cosGvR, cosGvR, Gpref)
        ewg02 += -numpy.einsum('gx,gy,ig,jg,g->ijxy', Gv,
                               Gv, sinGvR, sinGvR, Gpref)
        ewg02 /= 3

        return (ewself00 + ewg00)[util.atom_to_bas_indices_2d(self.mol)], \
            (ewself01 + ewg01)[util.atom_to_bas_indices(self.mol)], \
            (ewself11 + ewg11), \
            (ewself02 + ewg02)[util.atom_to_bas_indices(self.mol)]

    def get_qm_ewald_pot(self, mol, dm, qm_ewald_hess=None):
        # hess = d^2 E / dQ_i dQ_j, d^2 E / dQ_i dD_ja, d^2 E / dDia dDjb, d^2 E/ dQ_i dO_jab
        if qm_ewald_hess is None:
            if self.pbcqm:
                qm_ewald_hess = self.get_qm_ewald_hess()
            self.qm_ewald_hess = qm_ewald_hess
        charges = self.get_qm_charges(dm)
        if self.param.dipgam is not None:
            dips = self.get_qm_dipoles(dm)
        if self.param.quadgam is not None:
            quads = self.get_qm_quadrupoles(dm)
        return self.get_qm_ewald_pot_fromq(mol, self.pack_q(charges, dips, quads), qm_ewald_hess)

    def get_qm_ewald_pot_fromq(self, mol, q, qm_ewald_hess=None):
        if qm_ewald_hess is None:
            if self.pbcqm:
                qm_ewald_hess = self.get_qm_ewald_hess()
            self.qm_ewald_hess = qm_ewald_hess
        charges, dips, quads = self.unpack_q(q)
        ewpot0 = numpy.zeros_like(charges)
        ewpot1 = numpy.zeros((mol.natm, 3))
        ewpot2 = numpy.zeros((mol.natm, 3, 3))
        if self.pbcqm:
            ewpot0 = numpy.einsum('ij,j->i', qm_ewald_hess[0], charges)
            if self.param.dipgam is not None:
                ewpot0 += numpy.einsum('ijx,jx->i', qm_ewald_hess[1], dips)
                ewpot1 += numpy.einsum('ijx,i->jx', qm_ewald_hess[1], charges)
                ewpot1 += numpy.einsum('ijxy,jy->ix', qm_ewald_hess[2], dips)
            if self.param.quadgam is not None:
                ewpot0 += numpy.einsum('ijxy,jxy->i', qm_ewald_hess[3], quads)
                ewpot2 += numpy.einsum('ijxy,i->jxy',
                                       qm_ewald_hess[3], charges)
        else:
            pass
        return ewpot0, ewpot1, ewpot2

    def get_ovlp(self, *args):
        if self.s1 is None:
            self.s1 = self.mol.intor('int1e_ovlp', hermi=1)
        return self.s1

    def get_qm_charges(self, dm, s1e=None):
        ''' shell-resolved charges '''
        if s1e is None:
            s1e = self.get_ovlp()
        return mulliken_charge(self.mol, self.param, s1e, dm)

    def get_s1r(self):
        '''
        s1r[x,u,v] = <u|(rx - Rvx)|v>
        where Rv is center of AO v
        '''
        if self.s1r is None:
            self.s1r = list()
            mol = self.mol
            atm_to_ao_id = util.atom_to_ao_indices(mol)
            s1r = mol.intor('int1e_r', hermi=1)  # (3, nao, nao)
            self.s1r = s1r - numpy.einsum(
                'vx,uv->xuv',
                mol.atom_coords()[atm_to_ao_id],
                self.get_ovlp()
            )
        return self.s1r

    def get_qm_dipoles(self, dm, s1r=None):
        ''' atom-resolved dipoles '''
        if s1r is None:
            s1r = self.get_s1r()
        ao_dip = -numpy.einsum('uv,xvu->ux', dm, s1r)
        return numpy.zeros((self.mol.natm, 3)).at[
            util.atom_to_ao_indices(self.mol)].add(ao_dip)

    def get_s1rr(self):
        r'''
        \int phi_u phi_v [3(r-Rc)\otimes(r-Rc) - |r-Rc|^2] /2 dr
        '''
        if self.s1rr is None:
            mol = self.mol
            nao = mol.nao_nr()
            atm_to_ao_id = util.atom_to_ao_indices(mol)
            s1rr = mol.intor('int1e_rr', hermi=1).reshape(3, 3, nao, nao)
            s1r2 = numpy.einsum('xxuv->uv', s1rr)
            s1r = self.get_s1r()
            s1 = self.get_ovlp()
            Rv = mol.atom_coords()[atm_to_ao_id]
            self.s1rr = -1.5 * numpy.einsum('vx,yuv->xyuv', Rv, s1r)
            self.s1rr += self.s1rr.transpose([1, 0, 2, 3])
            self.s1rr += 1.5 * s1rr
            scalar = -0.5 * s1r2 + numpy.einsum('xuv,vx->uv', s1r, Rv) \
                + 0.5 * numpy.einsum('vx,vx,uv->uv', Rv, Rv, s1)
            self.s1rr += numpy.einsum('xy,uv->xyuv', numpy.eye(3), scalar)
            self.s1rr -= 1.5 * numpy.einsum('vx,vy,uv->xyuv', Rv, Rv, s1)
        return self.s1rr

    def get_qm_quadrupoles(self, dm, s1rr=None):
        ''' atom-resolved quadrupoles '''
        if s1rr is None:
            s1rr = self.get_s1rr()
        aoslices = self.mol.aoslice_by_atom()
        ao_quad = -numpy.einsum('uv,xyvu->uxy', dm, s1rr)
        return numpy.zeros((self.mol.natm, 3, 3)).at[
            util.atom_to_ao_indices(self.mol)
        ].add(ao_quad)

    def pack_q(self, mono, dip, quad):
        to_pack = [mono.ravel()]
        if self.param.dipgam is not None:
            to_pack.append(dip.ravel())
        if self.param.quadgam is not None:
            to_pack.append(quad.ravel())
        return numpy.concatenate(to_pack)

    def unpack_q(self, packed):
        nbas = self.mol.nbas
        natm = self.mol.natm
        mono = packed[:nbas]
        dip = quad = None
        if self.param.dipgam is not None:
            dip = packed[nbas:nbas+3*natm].reshape(natm, 3)
        if self.param.quadgam is not None:
            quad = packed[nbas+3*natm:].reshape(natm, 3, 3)
        return mono, dip, quad

    def get_q(self, mol=None, dm=None, s1e=None):
        if mol is None:
            mol = self.mol
        if dm is None:
            dm = self.make_rdm1()
        if s1e is None:
            s1e = self.get_ovlp()
        dip = quad = None
        if self.param.dipgam is not None:
            dip = self.get_qm_dipoles(dm)
        if self.param.quadgam is not None:
            quad = self.get_qm_quadrupoles(dm)
        return self.pack_q(self.get_qm_charges(dm, s1e=s1e), dip, quad)

    def get_vdiff(self, mol, ewald_pot):
        '''
        vdiff_uv = d Q_I / d dm_uv ewald_pot[0]_I
                 + d D_Ix / d dm_uv ewald_pot[1]_Ix 
                 + d O_Ixy / d dm_uv ewald_pot[2]_Ixy
        '''
        ovlp = self.get_ovlp()
        vdiff = -ewald_pot[0][util.bas_to_ao_indices(mol)] * ovlp
        atm_to_ao_id = util.atom_to_ao_indices(mol)
        if self.param.dipgam is not None:
            s1r = self.get_s1r()
            # vdiff[:,p0:p1] -= numpy.einsum('x,xuv->uv', v1, s1r[iatm])
            vdiff -= numpy.einsum(
                'vx,xuv->uv',
                ewald_pot[1][atm_to_ao_id],
                s1r
            )
        if self.param.quadgam is not None:
            s1rr = self.get_s1rr()
            # vdiff[:,p0:p1] -= numpy.einsum('xy,xyuv->uv', v2, s1rr[iatm])
            vdiff -= numpy.einsum(
                'vxy,xyuv->uv',
                ewald_pot[2][atm_to_ao_id],
                s1rr
            )
        vdiff = (vdiff + vdiff.T) / 2
        return numpy.asarray(vdiff)

    def get_veff(self, mol=None, dm=None, dm_last=0,
                 vhf_last=0, hermi=1, s1e=None, q=None,
                 mm_ewald_pot=None, qm_ewald_pot=None):
        del dm_last, vhf_last
        if mol is None:
            mol = self.mol
        if s1e is None:
            s1e = self.get_ovlp()
        if q is None:
            q = self.get_q(mol=mol, dm=dm, s1e=s1e)

        if mm_ewald_pot is None:
            if self.mm_ewald_pot is not None:
                mm_ewald_pot = self.mm_ewald_pot
            else:
                mm_ewald_pot = self.get_mm_ewald_pot(self.param)
                self.mm_ewald_pot = mm_ewald_pot

        if qm_ewald_pot is None:
            if self.qm_ewald_hess is not None:
                qm_ewald_pot = self.get_qm_ewald_pot_fromq(
                    mol, q, self.qm_ewald_hess)
            else:
                qm_ewald_pot = self.get_qm_ewald_pot_fromq(mol, q)

        ewald_pot = (mm_ewald_pot[0] + qm_ewald_pot[0],
                     mm_ewald_pot[1] + qm_ewald_pot[1],
                     mm_ewald_pot[2] + qm_ewald_pot[2])

        vdiff = self.get_vdiff(mol, ewald_pot)
        ediff = self.energy_ewald_fromq(
            q, mm_ewald_pot=mm_ewald_pot, qm_ewald_pot=qm_ewald_pot)

        veff = super().get_veff(mol=mol, s1e=s1e, q=q)
        return VXC(vxc=veff.vxc+vdiff, ecoul=veff.ecoul+ediff)

    def energy_ewald(self, dm=None, mm_ewald_pot=None, qm_ewald_pot=None):
        # QM-QM and QM-MM pbc energy
        if dm is None:
            dm = self.make_rdm1()
        if mm_ewald_pot is None:
            if self.mm_ewald_pot is not None:
                mm_ewald_pot = self.mm_ewald_pot
            else:
                mm_ewald_pot = self.get_mm_ewald_pot(self.param)
        if qm_ewald_pot is None:
            qm_ewald_pot = self.get_qm_ewald_pot(
                self.mol, dm, self.qm_ewald_hess)
        q = self.get_q(mol=self.mol, dm=dm)
        return self.energy_ewald_fromq(q, mm_ewald_pot, qm_ewald_pot)

    def energy_ewald_fromq(self, q, mm_ewald_pot=None, qm_ewald_pot=None):
        # QM-QM and QM-MM pbc energy
        if mm_ewald_pot is None:
            if self.mm_ewald_pot is not None:
                mm_ewald_pot = self.mm_ewald_pot
            else:
                mm_ewald_pot = self.get_mm_ewald_pot(self.param)
        if qm_ewald_pot is None:
            qm_ewald_pot = self.get_qm_ewald_pot_fromq(
                self.mol, q, self.qm_ewald_hess)
        ewald_pot = mm_ewald_pot[0] + qm_ewald_pot[0] / 2
        mono, dip, quad = self.unpack_q(q)
        e = numpy.einsum('i,i->', ewald_pot, mono)
        if self.param.dipgam is not None:
            ewald_pot = mm_ewald_pot[1] + qm_ewald_pot[1] / 2
            e += numpy.einsum('ix,ix->', ewald_pot, dip)
        if self.param.quadgam is not None:
            ewald_pot = mm_ewald_pot[2] + qm_ewald_pot[2] / 2
            e += numpy.einsum('ixy,ixy->', ewald_pot, quad)
        # energy correction for non zero total charge
        e += -.5 * numpy.sum(mono)**2 * numpy.pi/(self.qm_ew_eta**2 * self.vol)
        e += -1. * numpy.sum(mono) * \
            numpy.sum(self.mm_charges) * numpy.pi / \
            (self.mm_ew_eta**2 * self.vol)
        return e
