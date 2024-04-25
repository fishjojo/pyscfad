import sys
import h5py
import numpy
from jax import numpy as np
from pyscf import __config__
from pyscf.pbc.scf import khf as pyscf_khf
#from pyscfad import util
from pyscfad.lib import stop_grad, logger
from pyscfad.scf import hf as mol_hf
from pyscfad.pbc import df
from pyscfad.pbc.scf import hf as pbchf

# TODO add mo_coeff, which requires AD wrt complex numbers
#Traced_Attributes = ['cell', 'mo_energy', 'with_df']

def get_ovlp(mf, cell=None, kpts=None):
    if cell is None:
        cell = mf.cell
    if kpts is None:
        kpts = mf.kpts
    return pbchf.get_ovlp(cell, kpts)

def get_hcore(mf, cell=None, kpts=None):
    if cell is None:
        cell = mf.cell
    if kpts is None:
        kpts = mf.kpts
    if cell.pseudo:
        nuc = np.asarray(mf.with_df.get_pp(kpts))
    else:
        raise NotImplementedError
    if len(cell._ecpbas) > 0:
        raise NotImplementedError
    t = np.asarray(cell.pbc_intor('int1e_kin', comp=1, hermi=1, kpts=kpts))
    return nuc + t

def energy_elec(mf, dm_kpts=None, h1e_kpts=None, vhf_kpts=None):
    if dm_kpts is None:
        dm_kpts = mf.make_rdm1()
    if h1e_kpts is None:
        h1e_kpts = mf.get_hcore()
    if vhf_kpts is None:
        vhf_kpts = mf.get_veff(mf.cell, dm_kpts)

    nkpts = len(dm_kpts)
    e1 = 1./nkpts * np.einsum('kij,kji', dm_kpts, h1e_kpts)
    e_coul = 1./nkpts * np.einsum('kij,kji', dm_kpts, vhf_kpts) * 0.5
    mf.scf_summary['e1'] = stop_grad(e1.real)
    mf.scf_summary['e2'] = stop_grad(e_coul.real)
    logger.debug(mf, 'E1 = %s  E_coul = %s', e1, e_coul)
    if abs(e_coul.imag > mf.cell.precision*10):
        logger.warn(mf, 'Coulomb energy has imaginary part %s. '
                    'Coulomb integrals (e-e, e-N) may not converge !',
                    e_coul.imag)
    return (e1+e_coul).real, e_coul.real

def get_fock(mf, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, diis=None,
             diis_start_cycle=None, level_shift_factor=None, damp_factor=None):
    h1e_kpts, s_kpts, vhf_kpts, dm_kpts = h1e, s1e, vhf, dm
    if h1e_kpts is None:
        h1e_kpts = mf.get_hcore()
    if vhf_kpts is None:
        vhf_kpts = mf.get_veff(mf.cell, dm_kpts)
    vhf_kpts = getattr(vhf_kpts, 'vxc', vhf_kpts)
    f_kpts = h1e_kpts + vhf_kpts
    if cycle < 0 and diis is None:  # Not inside the SCF iteration
        return f_kpts

    if diis_start_cycle is None:
        diis_start_cycle = mf.diis_start_cycle
    if level_shift_factor is None:
        level_shift_factor = mf.level_shift
    if damp_factor is None:
        damp_factor = mf.damp
    if s_kpts is None:
        s_kpts = mf.get_ovlp()
    if dm_kpts is None:
        dm_kpts = mf.make_rdm1()

    if 0 <= cycle < diis_start_cycle-1 and abs(damp_factor) > 1e-4:
        f_kpts = [mol_hf.damping(s1e, dm_kpts[k] * 0.5, f_kpts[k], damp_factor)
                  for k, s1e in enumerate(s_kpts)]
    if diis and cycle >= diis_start_cycle:
        f_kpts = diis.update(s_kpts, dm_kpts, f_kpts, mf, h1e_kpts, vhf_kpts)
    if abs(level_shift_factor) > 1e-4:
        f_kpts = [mol_hf.level_shift(s, dm_kpts[k], f_kpts[k], level_shift_factor)
                  for k, s in enumerate(s_kpts)]
    return np.asarray(f_kpts)

def make_rdm1(mo_coeff_kpts, mo_occ_kpts, **kwargs):
    nkpts = len(mo_occ_kpts)
    dm = [mol_hf.make_rdm1(mo_coeff_kpts[k], mo_occ_kpts[k]) for k in range(nkpts)]
    return np.asarray(dm)

#@util.pytree_node(Traced_Attributes, num_args=1)
class KSCF(pbchf.SCF, pyscf_khf.KSCF):
    def __init__(self, cell, kpts=numpy.zeros((1,3)),
                 exxdiv=getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald')):#, **kwargs):
        if not cell._built:
            sys.stderr.write('Warning: cell.build() is not called in input\n')
            cell.build()
        self.cell = cell
        mol_hf.SCF.__init__(self, cell)

        self.with_df = df.FFTDF(cell, kpts=kpts)
        # Range separation JK builder
        self.rsjk = None

        self.exxdiv = exxdiv
        #self.kpts = kpts
        self.conv_tol = max(cell.precision * 10, 1e-8)

        self.exx_built = False
        self._keys = self._keys.union(['cell', 'exx_built', 'exxdiv', 'with_df', 'rsjk'])
        #self.__dict__.update(kwargs)

    def get_jk(self, cell=None, dm_kpts=None, hermi=1, kpts=None, kpts_band=None,
               with_j=True, with_k=True, omega=None, **kwargs):
        #if cell is None:
        #    cell = self.cell
        if kpts is None:
            kpts = self.kpts
        if dm_kpts is None:
            dm_kpts = self.make_rdm1()
        cpu0 = (logger.process_clock(), logger.perf_counter())
        if self.rsjk:
            raise NotImplementedError
        else:
            vj, vk = self.with_df.get_jk(dm_kpts, hermi, kpts, kpts_band,
                                         with_j, with_k, omega, self.exxdiv)
        logger.timer(self, 'vj and vk', *cpu0)
        return vj, vk

    def get_init_guess(self, cell=None, key='minao'):
        if cell is None:
            cell = self.cell
        dm0 = pyscf_khf.KSCF.get_init_guess(self, stop_grad(cell), key)
        dm0 = numpy.asarray(dm0)
        return dm0

    def eig(self, h_kpts, s_kpts):
        nkpts = len(h_kpts)
        eig_kpts = []
        mo_coeff_kpts = []

        for k in range(nkpts):
            e, c = self._eigh(h_kpts[k], s_kpts[k])
            eig_kpts.append(e)
            mo_coeff_kpts.append(c)
        return eig_kpts, mo_coeff_kpts

    def make_rdm1(self, mo_coeff_kpts=None, mo_occ_kpts=None, **kwargs):
        if mo_coeff_kpts is None:
            mo_coeff_kpts = self.mo_coeff
        if mo_occ_kpts is None:
            mo_occ_kpts = self.mo_occ
        return make_rdm1(mo_coeff_kpts, mo_occ_kpts, **kwargs)

    def dump_chk(self, envs):
        if self.chkfile:
            mol_hf.SCF.dump_chk(self, envs)
            with h5py.File(self.chkfile, 'a') as fh5:
                fh5['scf/kpts'] = stop_grad(self.kpts)
        return self

    get_hcore = get_hcore
    get_ovlp = get_ovlp
    get_fock = get_fock
    get_occ = pyscf_khf.KSCF.get_occ
    energy_elec = energy_elec
    get_fermi = pyscf_khf.KSCF.get_fermi

    get_veff = pyscf_khf.KSCF.get_veff
    get_j = pyscf_khf.KSCF.get_j
    get_k = pyscf_khf.KSCF.get_k
    get_grad = pyscf_khf.KSCF.get_grad

KRHF = KSCF
