from pyscfad import numpy, ops
from pyscfad.pbc.gto import cell
from pyscfad.xtb.xtb import mulliken_charge, util
from pyscfad.dft.rks import VXC
from pyscfad.lib import logger

from pyscfad.scipy.special import erf, erfc

from pyscf import lib, gto
from pyscf.gto.mole import is_au

import functools
import jax
from jax.lax import stop_gradient, scan
from typing import Tuple

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


def add_mm_charges(xtb_method, mm_coords, a, mm_charges, mm_radii, ewald_precision=1e-6, eta=None, mesh=None, pbcqm=True, unit=None):
    from pyscfad.ml.gto import MolePad
    if unit is None:
        unit = xtb_method.mol.unit
    if not is_au(unit):
        mm_coords = mm_coords / lib.param.BOHR
        a = a / lib.param.BOHR
        mm_radii = mm_radii / lib.param.BOHR
    xtbqmmm = QMMM(xtb_method,
                   mm_coords=mm_coords, a=a, mm_charges=mm_charges, mm_radii=mm_radii,
                   ewald_precision=ewald_precision,
                   eta=eta, mesh=mesh,
                   pbcqm=pbcqm,
                   )
    return lib.set_class(xtbqmmm, (QMMM, xtb_method.__class__))

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

class QMMM:
    def __init__(self, method, mm_coords, a, mm_charges, mm_radii, pbcqm=True, ewald_precision=1e-6, eta=None, mesh=None):
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

        self.ewald_precision = ewald_precision
        self.eta = eta
        self.mesh = mesh

        self.dimension = 3

    @property
    def vol(self):
        return abs(numpy.linalg.det(self.a))

    get_Gv_weights = cell.Cell.get_Gv_weights
    reciprocal_vectors = cell.Cell.reciprocal_vectors

    def lattice_vectors(self):
        return self.a

    def get_ewald_params(self, precision=None):
        if precision is None:
            precision = self.ewald_precision

        if self.eta is None:
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
            rcut = numpy.min(numpy.hstack([dist0, dist1]))

            e = precision
            Q = numpy.sum(self.mm_charges**2) + \
                numpy.sum(self.mol.atom_charges()**2)
            eta = stop_gradient(
                1 / rcut * numpy.sqrt(
                    1.5 *
                    lambertw(
                        2/3 * (4/e*Q/rcut/self.vol)**(2/3) * rcut**2
                    ).real
                )
            )
        else:
            eta = self.eta
        if self.mesh is None:
            L = self.vol**(1/3)
            kmax = 1.73205081*eta/2/numpy.pi * numpy.sqrt(
                lambertw(4*Q**(2/3)/3/numpy.pi**(2/3)/L**2/eta**(2/3) / e**(4/3)).real)
            mesh = stop_gradient(numpy.asarray(numpy.ceil(
                numpy.diag(self.a) * kmax) * 2 + 1, dtype=int))
        else:
            mesh = self.mesh

        return eta, mesh

    def get_mm_ewald_pot(self, param=None, chunk_size=1024):
        log = logger.new_logger(self, self.verbose)
        if param is None:
            param = self.param
        ew_eta, mesh = self.get_ewald_params()

        coords1 = self.mol.atom_coords()
        coords2 = self.mm_coords

        atom_to_bas = util.atom_to_bas_indices(self.mol)

        coord2_batches = _chunkize(coords2, chunk_size=chunk_size)
        mm_charge_batches = _chunkize(self.mm_charges, chunk_size=chunk_size)
        mm_radius_batches = _chunkize(self.mm_radii, chunk_size=chunk_size)

        @jax.checkpoint
        def accumulate_ewald_pot(carry, input):
            coords2, mm_charges, mm_radii = input
            ewovrl0, ewovrl1, ewovrl2 = carry

            R = coords1[:,None,:] - coords2[None,:,:]
            r2 = numpy.sum(R * R, axis=-1)
            r = numpy.sqrt(numpy.where(r2 < 1e-20, numpy.inf, r2))

            # difference between MM gaussain charges and MM point charges
            expnts = 2. / (1 / (param.gam*param.lgam)[:,None] + mm_radii[None])
            Tij = erfc(expnts * r[atom_to_bas]) / r[atom_to_bas]
            ewovrl0 -= numpy.einsum('ij,j->i', Tij, mm_charges)
            if param.dipgam is not None:
                expnts = 2. / (1 / param.dipgam[:,None] + mm_radii[None])
                ekR = numpy.exp(-expnts**2 * r**2)
                Tij = erfc(expnts * r) / r
                invr3 = (Tij + 2/numpy.sqrt(numpy.pi) * expnts * ekR) / r**2
                Tija = -numpy.einsum('ijx,ij->ijx', R, invr3)
                ewovrl1 -= numpy.einsum('j,ija->ia', mm_charges, Tija)
            if param.quadgam is not None:
                expnts = 2. / (1 / param.quadgam[:,None] + mm_radii[None])
                ekR = numpy.exp(-expnts**2 * r**2)
                Tij = erfc(expnts * r) / r
                invr3 = (Tij + 2/numpy.sqrt(numpy.pi) * expnts * ekR) / r**2
                Tija = -numpy.einsum('ijx,ij->ijx', R, invr3)
                Tijab  = 3 * numpy.einsum('ija,ijb,ij->ijab', R, R, 1/r**2)
                Tijab -= numpy.einsum('ij,ab->ijab', numpy.ones_like(r), numpy.eye(3))
                invr5 = invr3 + 4/3/numpy.sqrt(numpy.pi) * expnts**3 * ekR
                Tijab = numpy.einsum('ijab,ij->ijab', Tijab, invr5)
                Tijab += numpy.einsum('ij,ij,ab->ijab', expnts**3, 4/3/numpy.sqrt(numpy.pi)*ekR, numpy.eye(3))
                ewovrl2 -= numpy.einsum('j,ijab->iab', mm_charges, Tijab) / 3

            # ewald real-space sum; treat MM as point charges
            ekR = numpy.exp(-ew_eta**2 * r**2)
            # Tij = \hat{1/r} = f0 / r = erfc(r) / r
            Tij = erfc(ew_eta * r) / r
            ewovrl0 += numpy.einsum('ij,j->i', Tij, mm_charges)[atom_to_bas]
            if param.dipgam is not None:
                # Tija = -Rija \hat{1/r^3} = -Rija / r^2 ( \hat{1/r} + 2 eta/sqrt(pi) exp(-eta^2 r^2) )
                invr3 = (Tij + 2*ew_eta/numpy.sqrt(numpy.pi) * ekR) / r**2
                Tija = -numpy.einsum('ijx,ij->ijx', R, invr3)
                ewovrl1 += numpy.einsum('ijx,j->ix', Tija, mm_charges)
            if param.quadgam is not None:
                # Tijab = (3 RijaRijb - Rij^2 delta_ab) \hat{1/r^5}
                Tijab  = 3 * numpy.einsum('ija,ijb,ij->ijab', R, R, 1/r**2)
                Tijab -= numpy.einsum('ij,ab->ijab', numpy.ones_like(r), numpy.eye(3))
                invr5 = invr3 + 4/3*ew_eta**3/numpy.sqrt(numpy.pi) * ekR # NOTE this is invr5 * r**2
                Tijab = numpy.einsum('ijab,ij->ijab', Tijab, invr5)
                # NOTE the below is present in Eq 8 but missing in Eq 12
                Tijab += 4/3*ew_eta**3/numpy.sqrt(numpy.pi)*numpy.einsum('ij,ab->ijab', ekR, numpy.eye(3))
                ewovrl2 += numpy.einsum('ijxy,j->ixy', Tijab, mm_charges/3)

            return (ewovrl0, ewovrl1, ewovrl2), None

        (ewovrl0, ewovrl1, ewovrl2) = scan(
            accumulate_ewald_pot,
            (
                numpy.zeros_like(param.gam),
                numpy.zeros((len(coords1), 3)),
                numpy.zeros((len(coords1), 3, 3))
            ),
            (
                coord2_batches,
                mm_charge_batches,
                mm_radius_batches
            )
        )[0]
        cput1 = log.timer('MM Ewald Real-Space')

        # g-space sum (using g grid)
        Gv, Gvbase, weights = self.get_Gv_weights(mesh)
        absG2 = numpy.einsum('gx,gx->g', Gv, Gv)
        absG2 = numpy.where(absG2 == 0, 1e200, absG2)

        coulG = 4*numpy.pi / absG2
        coulG *= weights
        # NOTE Gpref is actually Gpref*2
        Gpref = numpy.exp(-absG2/(4*ew_eta**2)) * coulG

        zcosGvR2, zsinGvR2 = _structural_factor(
                Gv, coord2_batches, mm_charge_batches)

        GvR1 = numpy.einsum('gx,ix->ig', Gv, coords1)
        cosGvR1 = numpy.cos(GvR1)
        sinGvR1 = numpy.sin(GvR1)
        # qm pc - mm pc
        ewg0 = numpy.einsum('ig,g,g->i', cosGvR1, zcosGvR2, Gpref)
        ewg0 += numpy.einsum('ig,g,g->i', sinGvR1, zsinGvR2, Gpref)
        ewg0 = ewg0[atom_to_bas]
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

        cput1 = log.timer('MM Ewald G-Space', *cput1)
        del log
        return ewovrl0+ewg0, ewovrl1+ewg1, ewovrl2+ewg2

    def get_qm_ewald_hess(self):
        log = logger.new_logger(self)
        ew_eta, mesh = self.get_ewald_params()

        coords1 = self.mol.atom_coords()

        ewself00 = numpy.zeros((len(coords1), len(coords1)))
        ewself01 = numpy.zeros((len(coords1), len(coords1), 3))
        ewself11 = numpy.zeros((len(coords1), len(coords1), 3, 3))
        ewself02 = numpy.zeros((len(coords1), len(coords1), 3, 3))

        R = coords1[:,None] - coords1[None]
        r2 = numpy.sum(R * R , axis=-1)
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
        Tijab  = 3 * numpy.einsum('ija,ijb,ij->ijab', R, R, 1/r2)
        Tijab -= numpy.einsum('ij,ab->ijab', numpy.ones_like(r), numpy.eye(3))
        invr5 = invr3 + 4/3*ew_eta**3/numpy.sqrt(numpy.pi) * ekR # NOTE this is invr5 * r**2
        Tijab = numpy.einsum('ijab,ij->ijab', Tijab, invr5)
        # NOTE the below is present in Eq 8 but missing in Eq 12
        Tijab += 4/3*ew_eta**3/numpy.sqrt(numpy.pi)*numpy.einsum('ij,ab->ijab', ekR, numpy.eye(3))
        ewself00 += Tij
        ewself01 -= Tija
        ewself11 -= Tijab
        ewself02 += Tijab / 3

        # unit cell Coloumb, to be subtracted out
        Tij = 1 / r
        Tija = -numpy.einsum('ijx,ij->ijx', R, Tij**3)
        Tijab  = 3 * numpy.einsum('ija,ijb->ijab', R, R) 
        Tijab  = numpy.einsum('ijab,ij->ijab', Tijab, Tij**5)
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

        cput1 = log.timer('QM Ewald Real-Space')

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

        cput1 = log.timer('QM Ewald G-Space', *cput1)
        del log

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
            self.s1 = self.mol.intor('int1e_ovlp')
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
            log = logger.new_logger(self)
            self.s1r = list()
            mol = self.mol
            atm_to_ao_id = util.atom_to_ao_indices(mol)
            s1r = mol.intor('int1e_r')  # (3, nao, nao)
            self.s1r = s1r - numpy.einsum(
                'vx,uv->xuv',
                mol.atom_coords()[atm_to_ao_id],
                self.get_ovlp()
            )
            log.timer("get_s1r")
            del log
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
            log = logger.new_logger(self)
#            self.s1rr = list()
#            nao = mol.nao_nr()
#            aoslice = mol.aoslice_by_atom()
#            for i, c in zip(range(self.mol.natm), mol.atom_coords()):
#                b0, b1 = aoslice[i][:2]
#                shls_slice = (0, mol.nbas, b0, b1)
#                with mol.with_common_orig(c):
#                    s1rr_ = mol.intor('int1e_rr', shls_slice=shls_slice)
#                    s1rr_ = s1rr_.reshape((3, 3, nao, -1))
#                    s1rr_trace = numpy.einsum('xxuv->uv', s1rr_)
#                    s1rr_ = 3/2 * s1rr_
#                    for k in range(3):
#                        s1rr_ = s1rr_.at[k, k].subtract(0.5 * s1rr_trace)
#                self.s1rr.append(s1rr_)
            mol = self.mol
            nao = mol.nao_nr()
            atm_to_ao_id = util.atom_to_ao_indices(mol)
            s1rr = mol.intor('int1e_rr').reshape(3, 3, nao, nao)
            s1r2 = numpy.einsum('xxuv->uv', s1rr)
            s1r  = self.get_s1r()
            s1   = self.get_ovlp()
            Rv = mol.atom_coords()[atm_to_ao_id]
            self.s1rr  = -1.5 * numpy.einsum('vx,yuv->xyuv', Rv, s1r)
            self.s1rr += self.s1rr.transpose([1,0,2,3])
            self.s1rr +=  1.5 * s1rr
            scalar = -0.5 * s1r2 + numpy.einsum('xuv,vx->uv', s1r, Rv) \
                     -0.5 * numpy.einsum('vx,vx,uv->uv', Rv, Rv, s1)
            self.s1rr +=  0.5 * numpy.einsum('xy,uv->xyuv', numpy.eye(3), scalar)
            self.s1rr +=  1.5 * numpy.einsum('vx,vy,uv->xyuv', Rv, Rv, s1)
            log.timer("get_s1rr")
            del log
        return self.s1rr

    def get_qm_quadrupoles(self, dm, s1rr=None):
        ''' atom-resolved quadrupoles '''
        if s1rr is None:
            s1rr = self.get_s1rr()
        aoslices = self.mol.aoslice_by_atom()
        ao_quad = -numpy.einsum('uv,xyvu->uxy', dm, s1rr)
        return numpy.zeros((self.mol.natm,3,3)).at[
            util.atom_to_ao_indices(self.mol)
        ].add(ao_quad)
        qm_quadrupoles = list()
        for iatm in range(self.mol.natm):
            p0, p1 = aoslices[iatm, 2:]
            qm_quadrupoles.append(
                -numpy.einsum('uv,xyvu->xy', dm[p0:p1], s1rr[iatm]))
        return numpy.asarray(qm_quadrupoles)

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
        log = logger.new_logger(self)
        ovlp = self.get_ovlp()
        vdiff = -ewald_pot[0][util.bas_to_ao_indices(mol)] * ovlp
        if self.param.dipgam is not None:
            s1r = self.get_s1r()
        if self.param.quadgam is not None:
            s1rr = self.get_s1rr()
        atm_to_ao_id = util.atom_to_ao_indices(mol)
        if self.param.dipgam is not None:
            # vdiff[:,p0:p1] -= numpy.einsum('x,xuv->uv', v1, s1r[iatm])
            vdiff -= numpy.einsum(
                'vx,xuv->uv',
                ewald_pot[1][atm_to_ao_id],
                s1r
            )
        if self.param.quadgam is not None:
            # vdiff[:,p0:p1] -= numpy.einsum('xy,xyuv->uv', v2, s1rr[iatm])
            vdiff -= numpy.einsum(
                'vxy,xyuv->uv',
                ewald_pot[2][atm_to_ao_id],
                s1rr
            )
#        aoslices = mol.aoslice_by_atom()
#        for iatm in range(mol.natm):
##            v1 = ewald_pot[1][iatm]
#            v2 = ewald_pot[2][iatm]
#            p0, p1 = aoslices[iatm, 2:]
#            if self.param.quadgam is not None:
#                vdiff = vdiff.at[:, p0:p1].subtract(
#                    numpy.einsum('xy,xyuv->uv', v2, s1rr[iatm]))
        vdiff = (vdiff + vdiff.T) / 2
        log.timer("get_vdiff")
        del log
        return numpy.asarray(vdiff)

    def get_veff_fromq(self, q, mol=None, dm_last=0, vhf_last=0, hermi=1, s1e=None,
                       mm_ewald_pot=None, qm_ewald_pot=None):
        del dm_last, vhf_last
        if mol is None:
            mol = self.mol
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

        ewald_pot = \
            mm_ewald_pot[0] + qm_ewald_pot[0], \
            mm_ewald_pot[1] + qm_ewald_pot[1], \
            mm_ewald_pot[2] + qm_ewald_pot[2]
        vdiff = self.get_vdiff(mol, ewald_pot)
        ediff = self.energy_ewald_fromq(
            q, mm_ewald_pot=mm_ewald_pot, qm_ewald_pot=qm_ewald_pot)

        veff = super().get_veff_fromq(q, mol=mol, hermi=hermi, s1e=s1e)
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
        q = self.get_q(mol=mf.mol, dm=dm)
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
        eta, _ = self.get_ewald_params()
        e += -.5 * numpy.sum(mono)**2 * numpy.pi/(eta**2 * self.vol)
        e += -1. * numpy.sum(mono) * \
            numpy.sum(self.mm_charges) * numpy.pi/(eta**2 * self.vol)
        return e