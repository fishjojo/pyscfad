from functools import wraps
import numpy
from pyscf.lib import module_method
from pyscf.scf import uhf as pyscf_uhf
from pyscfad import numpy as np
from pyscfad import ops
from pyscfad.lib import logger
from pyscfad.scf import hf


@wraps(pyscf_uhf.get_fock)
def get_fock(mf, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, diis=None,
             diis_start_cycle=None, level_shift_factor=None, damp_factor=None,
             fock_last=None):
    if h1e is None:
        h1e = mf.get_hcore()
    if vhf is None:
        vhf = mf.get_veff(mf.mol, dm)
    f = np.asarray(h1e) + vhf
    if f.ndim == 2:
        f = (f, f)
    if cycle < 0 and diis is None:  # Not inside the SCF iteration
        return f

    if diis_start_cycle is None:
        diis_start_cycle = mf.diis_start_cycle
    if level_shift_factor is None:
        level_shift_factor = mf.level_shift
    if damp_factor is None:
        damp_factor = mf.damp
    if s1e is None:
        s1e = mf.get_ovlp()
    if dm is None:
        dm = mf.make_rdm1()

    if isinstance(level_shift_factor, (tuple, list, numpy.ndarray)):
        shifta, shiftb = level_shift_factor
    else:
        shifta = shiftb = level_shift_factor
    if isinstance(damp_factor, (tuple, list, numpy.ndarray)):
        dampa, dampb = damp_factor
    else:
        dampa = dampb = damp_factor

    if getattr(dm, 'ndim', None) == 2:
        dm = [dm*.5] * 2
    if 0 <= cycle < diis_start_cycle-1 and abs(dampa)+abs(dampb) > 1e-4 and fock_last is not None:
        f = (hf.damping(f[0], fock_last[0], dampa),
             hf.damping(f[1], fock_last[1], dampb))
    if diis and cycle >= diis_start_cycle:
        f = diis.update(s1e, dm, f, mf, h1e, vhf, f_prev=fock_last)
    if abs(shifta)+abs(shiftb) > 1e-4:
        f = (hf.level_shift(s1e, dm[0], f[0], shifta),
             hf.level_shift(s1e, dm[1], f[1], shiftb))
    return np.array(f)


@wraps(pyscf_uhf.energy_elec)
def energy_elec(mf, dm=None, h1e=None, vhf=None):
    if dm is None:
        dm = mf.make_rdm1()
    if h1e is None:
        h1e = mf.get_hcore()
    if getattr(dm, 'ndim', None) == 2:
        dm = np.array((dm*.5, dm*.5))
    if vhf is None:
        vhf = mf.get_veff(mf.mol, dm)
    if h1e[0].ndim < dm[0].ndim:  # get [0] because h1e and dm may not be ndarrays
        h1e = (h1e, h1e)
    e1 = np.einsum('ij,ji->', h1e[0], dm[0])
    e1+= np.einsum('ij,ji->', h1e[1], dm[1])
    e_coul =(np.einsum('ij,ji->', vhf[0], dm[0]) +
             np.einsum('ij,ji->', vhf[1], dm[1])) * .5
    e_elec = (e1 + e_coul).real
    mf.scf_summary['e1'] = e1.real
    mf.scf_summary['e2'] = e_coul.real
    logger.debug(mf, 'E1 = %s  Ecoul = %s', e1, e_coul.real)
    return e_elec, e_coul


@wraps(pyscf_uhf.make_rdm1)
def make_rdm1(mo_coeff, mo_occ, **kwargs):
    mo_a = mo_coeff[0]
    mo_b = mo_coeff[1]

    dm_a = np.dot(mo_a*mo_occ[0], mo_a.conj().T)
    dm_b = np.dot(mo_b*mo_occ[1], mo_b.conj().T)
    return np.array((dm_a, dm_b))


@wraps(pyscf_uhf.get_grad)
def get_grad(mo_coeff, mo_occ, fock_ao):
    occidxa = mo_occ[0] > 0
    occidxb = mo_occ[1] > 0
    viridxa = ~occidxa
    viridxb = ~occidxb

    ga = mo_coeff[0][:,viridxa].conj().T.dot(fock_ao[0].dot(mo_coeff[0][:,occidxa]))
    gb = mo_coeff[1][:,viridxb].conj().T.dot(fock_ao[1].dot(mo_coeff[1][:,occidxb]))
    return np.hstack((ga.ravel(), gb.ravel()))


class UHF(hf.SCF, pyscf_uhf.UHF):
    def __init__(self, mol):
        pyscf_uhf.UHF.__init__(self, mol)

    def eig(self, h, s):
        e_a, c_a = self._eigh(h[0], s)
        e_b, c_b = self._eigh(h[1], s)
        return np.array((e_a,e_b)), np.array((c_a,c_b))

    @wraps(pyscf_uhf.UHF.get_veff)
    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        if mol is None:
            mol = self.mol
        if dm is None:
            dm = self.make_rdm1()
        if getattr(dm, 'ndim', None) == 2:
            dm = np.asarray((dm*.5,dm*.5))
        if self._eri is not None or not self.direct_scf:
            vj, vk = self.get_jk(mol, dm, hermi)
            vhf = vj[0] + vj[1] - vk
        else:
            ddm = np.asarray(dm) - np.asarray(dm_last)
            vj, vk = self.get_jk(mol, ddm, hermi)
            vhf = vj[0] + vj[1] - vk
            vhf += np.asarray(vhf_last)
        return vhf

    def get_occ(self, mo_energy=None, mo_coeff=None):
        if mo_energy is None:
            mo_energy = self.mo_energy
        mo_energy = ops.to_numpy(mo_energy)
        if mo_coeff is not None:
            mo_coeff = ops.to_numpy(mo_coeff)
        return pyscf_uhf.UHF.get_occ(self, mo_energy, mo_coeff)

    def get_grad(self, mo_coeff, mo_occ, fock=None):
        if fock is None:
            dm1 = self.make_rdm1(mo_coeff, mo_occ)
            fock = self.get_hcore(self.mol) + self.get_veff(self.mol, dm1)
        return get_grad(mo_coeff, mo_occ, fock)

    spin_square = ops.stop_trace(pyscf_uhf.UHF.spin_square)
    get_fock = get_fock
    make_rdm1 = module_method(make_rdm1, absences=['mo_coeff', 'mo_occ'])
    energy_elec = energy_elec

