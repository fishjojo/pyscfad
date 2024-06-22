from functools import reduce, wraps
import numpy
from pyscf.scf import rohf as pyscf_rohf
from pyscfad import numpy as np
from pyscfad import util
from pyscfad.ops import stop_grad
from pyscfad.lib import logger
from pyscfad.scf import hf, uhf, chkfile

@wraps(pyscf_rohf.energy_elec)
def energy_elec(mf, dm=None, h1e=None, vhf=None):
    if dm is None:
        dm = mf.make_rdm1()
    elif getattr(dm, 'ndim', None) == 2:
        dm = np.array((dm*.5, dm*.5))
    return uhf.energy_elec(mf, dm, h1e, vhf)

@util.pytree_node(['fock', 'focka', 'fockb'], num_args=1)
class _FockMatrix():
    def __init__(self, fock, focka=None, fockb=None):
        self.fock = fock
        self.focka = focka
        self.fockb = fockb

    def __repr__(self):
        return self.fock.__repr__()

@util.pytree_node(['mo_energy',], num_args=1)
class _OrbitalEnergy():
    def __init__(self, mo_energy, mo_ea=None, mo_eb=None):
        self.mo_energy = mo_energy
        self.mo_ea = mo_ea
        self.mo_eb = mo_eb

    def __repr__(self):
        return self.mo_energy.__repr__()

def get_fock(mf, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, diis=None,
             diis_start_cycle=None, level_shift_factor=None, damp_factor=None):
    if h1e is None: h1e = mf.get_hcore()
    if s1e is None: s1e = mf.get_ovlp()
    if vhf is None: vhf = mf.get_veff(mf.mol, dm)
    if dm is None: dm = mf.make_rdm1()
    if getattr(dm, 'ndim', None) == 2:
        dm = np.array((dm*.5, dm*.5))
# To Get orbital energy in get_occ, we saved alpha and beta fock, because
# Roothaan effective Fock cannot provide correct orbital energy with `eig`
# TODO, check other treatment  J. Chem. Phys. 133, 141102
    focka = h1e + vhf[0]
    fockb = h1e + vhf[1]
    f = get_roothaan_fock((focka,fockb), dm, s1e)
    if cycle < 0 and diis is None:  # Not inside the SCF iteration
        return _FockMatrix(f, focka, fockb)

    if diis_start_cycle is None:
        diis_start_cycle = mf.diis_start_cycle
    if level_shift_factor is None:
        level_shift_factor = mf.level_shift
    if damp_factor is None:
        damp_factor = mf.damp

    dm_tot = dm[0] + dm[1]
    if 0 <= cycle < diis_start_cycle-1 and abs(damp_factor) > 1e-4:
        raise NotImplementedError('ROHF Fock-damping')
    if diis and cycle >= diis_start_cycle:
        f = diis.update(s1e, dm_tot, f, mf, h1e, vhf)
    if abs(level_shift_factor) > 1e-4:
        f = hf.level_shift(s1e, dm_tot*.5, f, level_shift_factor)
    return _FockMatrix(f, focka, fockb)

def get_roothaan_fock(focka_fockb, dma_dmb, s):
    nao = s.shape[0]
    focka, fockb = focka_fockb
    dma, dmb = dma_dmb
    fc = (focka + fockb) * .5
# Projector for core, open-shell, and virtual
    pc = np.dot(dmb, s)
    po = np.dot(dma-dmb, s)
    pv = np.eye(nao) - np.dot(dma, s)
    fock  = reduce(np.dot, (pc.conj().T, fc, pc)) * .5
    fock += reduce(np.dot, (po.conj().T, fc, po)) * .5
    fock += reduce(np.dot, (pv.conj().T, fc, pv)) * .5
    fock += reduce(np.dot, (po.conj().T, fockb, pc))
    fock += reduce(np.dot, (po.conj().T, focka, pv))
    fock += reduce(np.dot, (pv.conj().T, fc, pc))
    fock = fock + fock.conj().T
    return fock

def get_grad(mo_coeff, mo_occ, fock):
    occidxa = mo_occ > 0
    occidxb = mo_occ == 2
    viridxa = ~occidxa
    viridxb = ~occidxb
    uniq_var_a = viridxa.reshape(-1,1) & occidxa
    uniq_var_b = viridxb.reshape(-1,1) & occidxb

    if getattr(fock, 'focka', None) is not None:
        focka = fock.focka
        fockb = fock.fockb
    elif isinstance(fock, (tuple, list)) or getattr(fock, 'ndim', None) == 3:
        focka, fockb = fock
    else:
        focka = fockb = fock
    focka = reduce(numpy.dot, (mo_coeff.conj().T, focka, mo_coeff))
    fockb = reduce(numpy.dot, (mo_coeff.conj().T, fockb, mo_coeff))

    g = numpy.zeros_like(focka)
    g[uniq_var_a]  = focka[uniq_var_a]
    g[uniq_var_b] += fockb[uniq_var_b]
    return g[uniq_var_a | uniq_var_b].ravel()

def make_rdm1(mo_coeff, mo_occ, **kwargs):
    if getattr(mo_occ, 'ndim', None) == 1:
        mo_occa = mo_occ > 0
        mo_occb = mo_occ == 2
    else:
        mo_occa, mo_occb = mo_occ
    dm_a = np.dot(mo_coeff*mo_occa, mo_coeff.conj().T)
    dm_b = np.dot(mo_coeff*mo_occb, mo_coeff.conj().T)
    return np.array((dm_a, dm_b))

def get_occ(mf, mo_energy=None, mo_coeff=None):
    from pyscf.scf.rohf import _fill_rohf_occ
    if mo_energy is None: mo_energy = mf.mo_energy
    if getattr(mo_energy, 'mo_ea', None) is not None:
        mo_ea = numpy.asarray(mo_energy.mo_ea)
        mo_eb = numpy.asarray(mo_energy.mo_eb)
        mo_eab = numpy.asarray(mo_energy.mo_energy)
    else:
        mo_ea = mo_eb = mo_eab = numpy.asarray(mo_energy)
    nmo = mo_ea.size
    mo_occ = numpy.zeros(nmo)
    if getattr(mf, 'nelec', None) is None:
        nelec = mf.mol.nelec
    else:
        nelec = mf.nelec
    if nelec[0] > nelec[1]:
        nocc, ncore = nelec
    else:
        ncore, nocc = nelec
    nopen = nocc - ncore
    mo_occ = _fill_rohf_occ(mo_eab, mo_ea, mo_eb, ncore, nopen)

    if mf.verbose >= logger.INFO and nocc < nmo and ncore > 0:
        ehomo = max(mo_eab[mo_occ> 0])
        elumo = min(mo_eab[mo_occ==0])
        if ehomo+1e-3 > elumo:
            logger.warn(mf, 'HOMO %.15g >= LUMO %.15g', ehomo, elumo)
        else:
            logger.info(mf, '  HOMO = %.15g  LUMO = %.15g', ehomo, elumo)
        if nopen > 0 and mf.verbose >= logger.DEBUG:
            core_idx = mo_occ == 2
            open_idx = mo_occ == 1
            vir_idx = mo_occ == 0
            logger.debug(mf, '                  Roothaan           | alpha              | beta')
            logger.debug(mf, '  Highest 2-occ = %18.15g | %18.15g | %18.15g',
                         max(mo_eab[core_idx]),
                         max(mo_ea[core_idx]), max(mo_eb[core_idx]))
            logger.debug(mf, '  Lowest 0-occ =  %18.15g | %18.15g | %18.15g',
                         min(mo_eab[vir_idx]),
                         min(mo_ea[vir_idx]), min(mo_eb[vir_idx]))
            for i in numpy.where(open_idx)[0]:
                logger.debug(mf, '  1-occ =         %18.15g | %18.15g | %18.15g',
                             mo_eab[i], mo_ea[i], mo_eb[i])

        if mf.verbose >= logger.DEBUG:
            numpy.set_printoptions(threshold=nmo)
            logger.debug(mf, '  Roothaan mo_energy =\n%s', mo_eab)
            logger.debug1(mf, '  alpha mo_energy =\n%s', mo_ea)
            logger.debug1(mf, '  beta  mo_energy =\n%s', mo_eb)
            numpy.set_printoptions(threshold=1000)
    return mo_occ

@util.pytree_node(hf.Traced_Attributes, num_args=1)
class ROHF(hf.SCF, pyscf_rohf.ROHF):
    def __init__(self, mol, **kwargs):
        super().__init__(mol)
        self.__dict__.update(kwargs)

    def eig(self, fock, s):
        focka = getattr(fock, 'focka', None)
        fockb = getattr(fock, 'fockb', None)
        fockab = getattr(fock, 'fock', fock)
        e, c = self._eigh(fockab, s)
        if focka is not None:
            c_copy = numpy.asarray(stop_grad(c))
            focka_copy = numpy.asarray(stop_grad(focka))
            fockb_copy = numpy.asarray(stop_grad(fockb))
            mo_ea = numpy.einsum('pi,pi->i', c_copy.conj(), focka_copy.dot(c_copy)).real
            mo_eb = numpy.einsum('pi,pi->i', c_copy.conj(), fockb_copy.dot(c_copy)).real
            e = _OrbitalEnergy(e, mo_ea, mo_eb)
        return e, c

    def get_grad(self, mo_coeff, mo_occ, fock=None):
        if fock is None:
            dm1 = self.make_rdm1(mo_coeff, mo_occ)
            fock = self.get_hcore(self.mol) + self.get_veff(self.mol, dm1)
        return get_grad(stop_grad(mo_coeff), stop_grad(mo_occ), stop_grad(fock))

    def make_rdm1(self, mo_coeff=None, mo_occ=None, **kwargs):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if mo_occ is None: mo_occ = self.mo_occ
        if self.mol.spin < 0:
            # Flip occupancies of alpha and beta orbitals
            mo_occ = (mo_occ == 2), (mo_occ > 0)
        return make_rdm1(mo_coeff, mo_occ, **kwargs)

    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        if getattr(dm, 'ndim', None) == 2:
            dm = np.array((dm*.5, dm*.5))

        if self._eri is not None or not self.direct_scf:
            vj, vk = self.get_jk(mol, dm, hermi)
            vhf = vj[0] + vj[1] - vk
        else:
            ddm = dm - np.asarray(dm_last)
            vj, vk = self.get_jk(mol, ddm, hermi)
            vhf = vj[0] + vj[1] - vk
            vhf += np.asarray(vhf_last)
        return vhf

    def dump_chk(self, envs):
        if self.chkfile:
            mo_energy = getattr(envs['mo_energy'], 'mo_energy',
                                envs['mo_energy'])
            chkfile.dump_scf(self.mol, self.chkfile,
                             envs['e_tot'], mo_energy,
                             envs['mo_coeff'], envs['mo_occ'],
                             overwrite_mol=False)
        return self

    get_occ = get_occ
    get_fock = get_fock
    energy_elec = energy_elec
