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

"""
XTB with k-point sampling
"""
import numpy
from jax.scipy.special import erfc

from pyscfad import numpy as np
from pyscfad import ops
from pyscfad.ops import jit
from pyscfad.lib import logger
from pyscfad.gto.mole import intor_cross, inter_distance
from pyscfad.pbc.gto.cell import shift_bas_center
from pyscfad.pbc.scf import khf
from pyscfad.dft.rks import VXC

from pyscfad.xtb import util
from pyscfad.xtb import xtb
from pyscfad.xtb.data.radii import ATOMIC as ATOMIC_RADII


@jit
def EHT_PI_GFN1(cell, param, atomic_radii=ATOMIC_RADII,
                Ls=numpy.zeros((1,3))):
    shpoly = param.shpoly

    rr = inter_distance(cell, Ls=Ls)
    #rr = np.where(rr>1e-6, rr, 0)

    z = cell.atom_charges()
    cov_radii = atomic_radii[z]
    RAB = cov_radii[:,None] + cov_radii[None,:]
    #rr = np.where(rr>1e-6, np.sqrt(rr / RAB), 0)
    rr = np.sqrt(rr / RAB[None,:,:])

    i, j = util.atom_to_bas_indices_2d(cell)
    RR = rr[:,i,j]
    PI = (1 + shpoly[None,:,None] * RR) * (1 + shpoly[None,None,:] * RR)
    return PI

@jit
def mulliken_charge(cell, param, s1e, dm):
    s1e = np.asarray(s1e)
    dm = np.asarray(dm)
    nkpts = s1e.shape[0]

    SP = np.einsum("kpq,kpq->p", s1e, dm).real
    occs = np.zeros(cell.nbas)
    occs = ops.index_add(occs, ops.index[util.bas_to_ao_indices(cell)], SP)

    occs /= nkpts
    return param.refocc - occs

def estimate_ke_ewald(a, q, prec):
    q = max(q, 0.01)
    fac = prec / (32 * numpy.pi**2 * q * a**2)
    G = 1.
    G = numpy.sqrt(-numpy.log(fac * G)) * 2 * a
    G = numpy.sqrt(-numpy.log(fac * G)) * 2 * a
    return G

@jit
def _r_and_inv_r(cell, coords=None, Ls=None):
    r = inter_distance(cell, coords=coords, Ls=Ls)
    r_inv = 1. / np.where(r>1e-6, r, np.inf)
    return r, r_inv

def gamma_GFN1(cell, param, rcut=None):
    gamma_sr = init_gamma_sr(cell, param, rcut=rcut)
    gamma_ewald = init_gamma_ewald(cell)
    gamma = gamma_sr + gamma_ewald
    return gamma

def init_gamma_sr(cell, param, charge=1., rcut=None):
    eta = 1. / (param.lgam * param.gam)
    eta = 2. / (eta[:,None] + eta[None,:])
    alpha = .5 * numpy.sqrt(numpy.pi) * eta

    if rcut is None:
        rcut = numpy.max(util.rcut_erfc(alpha, charge, cell.precision))
    Ls = cell.get_lattice_Ls(rcut=rcut)

    r, r_inv = _r_and_inv_r(cell, Ls=Ls)

    i, j = util.atom_to_bas_indices_2d(cell)
    r = r[:,i,j]
    r_inv = r_inv[:,i,j]

    gamma = np.where(r<1e-6, eta[None,:,:], -erfc(alpha[None,:,:] * r) * r_inv)
    phi = np.sum(gamma, axis=0)
    return phi

def init_gamma_ewald(cell, charge=1.):
    # 1/r
    ew_eta = cell.get_ewald_params()[0]
    ew_cut = numpy.max(util.rcut_erfc(ew_eta, charge, cell.precision))

    Ls = cell.get_lattice_Ls(rcut=ew_cut)

    r, r_inv = _r_and_inv_r(cell, Ls=Ls)

    phi = np.sum(erfc(ew_eta * r) * r_inv, axis=0)

    ke_cutoff = estimate_ke_ewald(ew_eta, charge,
                                  cell.precision * ops.to_numpy(cell.vol))
    mesh = cell.cutoff_to_mesh(ke_cutoff)
    Gv, _, weights = cell.get_Gv_weights(mesh)
    absG2 = np.einsum('gi,gi->g', Gv, Gv)
    absG2 = np.where(absG2==0., np.inf, absG2)

    coulG = 4 * numpy.pi / absG2
    coulG *= weights

    SI = np.exp(-1j * (cell.atom_coords() @ Gv.T))
    expG2 = SI * np.exp(-absG2 / (4 * ew_eta**2))
    phi += np.einsum("ag,cg,g->ac", SI.conj(), expG2, coulG).real

    i, j = util.atom_to_bas_indices_2d(cell)
    phi = phi[i,j]
    return phi

@jit
def get_veff(mf, cell=None, dm=None, s1e=None):
    if cell is None:
        cell = mf.cell
    if dm is None:
        dm = mf.make_rdm1()
    if s1e is None:
        s1e = mf.get_ovlp()

    param = mf.param
    partial_charge = mulliken_charge(cell, param, s1e, dm)

    phi = np.dot(mf.gamma, partial_charge)

    atm_charge = xtb.sum_shell_charges(cell, partial_charge)

    atm_to_bas_id = util.atom_to_bas_indices(cell)
    ew_eta = mf.ew_eta
    phi += -2 * ew_eta / numpy.sqrt(numpy.pi) * atm_charge[atm_to_bas_id]

    ecoul = .5 * np.dot(partial_charge, phi)

    # FIXME assume charge neutral
    #ecoul += -.5 * np.pi/(ew_eta**2 * cell.vol) * np.sum(partial_charge)**2
    #phi += -np.pi/(ew_eta**2 * cell.vol) * np.sum(partial_charge)

    # Third-order term
    phi3 = atm_charge**2 * param.gam3
    ecoul += np.sum(atm_charge**3 * param.gam3) / 3.

    phi += phi3[atm_to_bas_id]
    phi = phi[util.bas_to_ao_indices(cell)]
    phi = phi[:,None] + phi[None,:]

    vj = -.5 * s1e * phi[None,...]
    vxc = VXC(vxc=vj, ecoul=ecoul)
    return vxc

def energy_nuc(mf, cell=None, rcut=None):
    log = logger.new_logger(mf)
    cput0 = log.get_t0()

    if cell is None:
        cell = mf.cell

    param = mf.param

    kf = param.kf
    zeff = param.zeff
    arep = param.arep

    if rcut is None:
        rcut = numpy.max(rcut_erep(cell, kf, zeff, arep))
    Ls = cell.get_lattice_Ls(cell, rcut=rcut)
    r, r_inv = _r_and_inv_r(cell, Ls=Ls)

    z_ab = zeff[:,None] * zeff[None,:]
    arep_ab = np.sqrt(arep[:,None] * arep[None,:])
    enuc = .5 * np.sum(z_ab[None,...] * np.exp(-arep_ab[None,...] * r**kf) * r_inv)

    log.timer("energy_nuc", *cput0)
    del log
    return enuc

def rcut_erep(cell, kf, zeff, arep, extra_prec=1):
    kf = ops.to_numpy(kf)
    zeff = ops.to_numpy(zeff)
    arep = ops.to_numpy(arep)

    prec = cell.precision * extra_prec

    r = 20.
    r = (-numpy.log(r * prec / zeff**2) / arep) ** (1/kf)
    r = (-numpy.log(r * prec / zeff**2) / arep) ** (1/kf)
    return r 


def get_occ(mf, mo_energy_kpts=None, mo_coeff_kpts=None):
    if mo_energy_kpts is None:
        mo_energy_kpts = mf.mo_energy

    nocc = mf.tot_electrons // 2

    mo_energy = np.sort(np.hstack(mo_energy_kpts))
    fermi = mo_energy[nocc-1]
    mo_occ_kpts = []
    for mo_e in mo_energy_kpts:
        mo_occ_kpts.append((mo_e <= fermi).astype(float) * 2)

    if nocc < mo_energy.size:
        logger.info(mf, 'HOMO = %.12g  LUMO = %.12g',
                    mo_energy[nocc-1], mo_energy[nocc])
        if mo_energy[nocc-1]+1e-3 > mo_energy[nocc]:
            logger.warn(mf, 'HOMO %.12g == LUMO %.12g',
                        mo_energy[nocc-1], mo_energy[nocc])
    else:
        logger.info(mf, 'HOMO = %.12g', mo_energy[nocc-1])

    if mf.verbose >= logger.DEBUG:
        numpy.set_printoptions(threshold=len(mo_energy))
        logger.debug(mf, '     k-point                  mo_energy')
        for k,kpt in enumerate(mf.cell.get_scaled_kpts(mf.kpts)):
            logger.debug(mf, '  %2d (%6.3f %6.3f %6.3f)   %s %s',
                         k, kpt[0], kpt[1], kpt[2],
                         np.sort(mo_energy_kpts[k][mo_occ_kpts[k]> 0]),
                         np.sort(mo_energy_kpts[k][mo_occ_kpts[k]==0]))
        numpy.set_printoptions(threshold=1000)

    return mo_occ_kpts


class KXTB(xtb.XTB):
    """Base class for XTB with k-point sampling
    """
    def __init__(self, cell, param,
                 kpts=numpy.zeros((1,3)), rcut=None):
        super().__init__(cell, param)
        self.kpts = kpts
        self.rcut = rcut

    def get_ovlp(self, cell=None, kpts=None):
        if cell is None:
            cell = self.cell
        if kpts is None:
            kpts = self.kpts
        s = cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts)
        return np.asarray(s)

    def energy_elec(self, dm=None, h1e=None, vhf=None):
        if dm is None:
            dm = self.make_rdm1()
        if h1e is None:
            h1e = mf.get_hcore(cell=self.cell, kpts=self.kpts)
        if vhf is None or getattr(vhf, 'ecoul', None) is None:
            vhf = mf.get_veff(self.cell, dm)

        weight = 1. / len(h1e)
        e1 = weight * np.einsum('kij,kji', h1e, dm)
        ecoul = vhf.ecoul
        tot_e = e1 + ecoul
        return tot_e.real, ecoul

    @property
    def cell(self):
        return self.mol

    @property
    def tot_electrons(self):
        return xtb.tot_valence_electrons(self.mol, nkpts=len(self.kpts))

    get_occ = get_occ
    make_rdm1 = khf.KSCF.make_rdm1
    eig = khf.KSCF.eig
    get_grad = khf.KSCF.get_grad


class GFN1KXTB(KXTB):
    """GFN1-XTB with k-point sampling
    """
    def __init__(self, cell, param,
                 kpts=numpy.zeros((1,3)), rcut=None):
        super().__init__(cell, param, kpts=kpts, rcut=rcut)
        _ = self.gamma
        self.ew_eta = cell.get_ewald_params()[0]

    def _energy_nuc(self, cell=None, rcut=None):
        if cell is None:
            cell = self.cell
        if rcut is None:
            rcut = self.rcut
        return energy_nuc(self, cell=cell, rcut=rcut)

    def get_init_guess(self, cell=None, key='refocc', s1e=None, **kwargs):
        if cell is None:
            cell = self.cell
        if s1e is None:
            s1e = self.get_ovlp(cell)

        dm = xtb.GFN1XTB.get_init_guess(self, cell, key)

        nkpts = len(self.kpts)
        dm_kpts = np.repeat(dm[None,:,:], nkpts, axis=0)

        ne = np.einsum('kij,kji->', dm_kpts, s1e).real
        nelectron = self.tot_electrons
        if abs(ne - nelectron) > 0.01*nkpts:
            logger.debug(self, 'Big error detected in the electron number '
                         'of initial guess density matrix (Ne/cell = %g)!\n'
                         '  This can cause huge error in Fock matrix and '
                         'lead to instability in SCF for low-dimensional '
                         'systems.\n  DM is normalized wrt the number '
                         'of electrons %s', ne/nkpts, nelectron/nkpts)
            dm_kpts *= (nelectron / ne).reshape(-1,1,1)
        return dm_kpts

    def get_hcore(self, cell=None, s1e=None, kpts=None):
        if cell is None:
            cell = self.cell
        if kpts is None:
            kpts = self.kpts

        log = logger.new_logger(self)
        cput0 = log.get_t0()

        param = self.param

        mask = util.mask_valence_shell_gfn1(cell)
        hscale = np.where(numpy.outer(mask, mask),
                          param.k_shlpr * param.kpair * xtb.EHT_X_GFN1(cell, param),
                          param.k_shlpr)

        hdiag = xtb.EHT_Hdiag_GFN1(cell, param)
        mask = util.mask_atom_pairs(cell)[util.atom_to_bas_indices_2d(cell)]

        # FIXME this rcut is too tight for low precision calculations
        rcut = cell.rcut_by_shells(precision=cell.precision**2*1e3).max()
        Ls = cell.get_lattice_Ls(rcut=rcut)
        nL = len(Ls)
        h1 = np.where(np.repeat(mask[None,:,:], nL, axis=0),
                      hscale[None,:,:] * EHT_PI_GFN1(cell, param, Ls=Ls) * hdiag[None,:,:],
                      np.repeat(hdiag[None,:,:], nL, axis=0))

        # TODO optimize lattice sum
        ss = []
        for L in Ls:
            shifted_cell = shift_bas_center(cell, L)
            ss.append(intor_cross("int1e_ovlp", cell, shifted_cell))
        ss = np.asarray(ss)

        expkL = np.exp(1j*np.dot(kpts, Ls.T))
        i, j = util.bas_to_ao_indices_2d(cell)
        hcore = np.einsum("kl,lpq->kpq", expkL, ss * h1[:,i,j])

        log.timer("get_hcore", *cput0)
        del log
        return hcore

    def get_veff(self, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1,
                 kpts=None, kpts_band=None, s1e=None):
        if cell is None:
            cell = self.cell
        if dm is None:
            dm = self.make_rdm1()
        if kpts is None:
            kpts = self.kpts
        if s1e is None:
            s1e = mf.get_ovlp(kpts=kpts)

        log = logger.new_logger(self)
        cput0 = log.get_t0()

        veff = get_veff(self, cell=cell, dm=dm, s1e=s1e)

        log.timer("get_veff", *cput0)
        del log
        return veff

    def _get_gamma(self):
        return gamma_GFN1(self.cell, self.param, rcut=self.rcut)
