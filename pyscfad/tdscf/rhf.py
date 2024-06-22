from functools import reduce
import numpy
from pyscf import symm
from pyscf.scf import hf_symm
from pyscf.tdscf import rhf as pyscf_tdrhf
from pyscfad import numpy as np
from pyscfad import ops
from pyscfad.ops import vmap, stop_grad
from pyscfad import util
from pyscfad.lib import logger, chkfile
from pyscfad.lib.linalg_helper import davidson1
from pyscfad import ao2mo
from pyscfad.gto import mole

Traced_Attributes = ['_scf', 'mol']

def gen_tda_operation(mf, fock_ao=None, singlet=True, wfnsym=None):
    mol = mf.mol
    mo_coeff = mf.mo_coeff
    #assert (mo_coeff.dtype == numpy.double)
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    nao, nmo = mo_coeff.shape
    occidx = numpy.where(mo_occ==2)[0]
    viridx = numpy.where(mo_occ==0)[0]
    nocc = len(occidx)
    nvir = len(viridx)
    orbv = mo_coeff[:,viridx]
    orbo = mo_coeff[:,occidx]

    if wfnsym is not None and mol.symmetry:
        if isinstance(wfnsym, str):
            wfnsym = symm.irrep_name2id(mol.groupname, wfnsym)
        wfnsym = wfnsym % 10  # convert to D2h subgroup
        orbsym = hf_symm.get_orbsym(mol, mo_coeff)
        orbsym_in_d2h = numpy.asarray(orbsym) % 10  # convert to D2h irreps
        sym_forbid = (orbsym_in_d2h[occidx,None] ^ orbsym_in_d2h[viridx]) != wfnsym

    if fock_ao is None:
        foo = np.diag(mo_energy[occidx])
        fvv = np.diag(mo_energy[viridx])
    else:
        fock = reduce(np.dot, (mo_coeff.conj().T, fock_ao, mo_coeff))
        foo = fock[occidx[:,None],occidx]
        fvv = fock[viridx[:,None],viridx]

    hdiag = fvv.diagonal() - foo.diagonal()[:,None]
    if wfnsym is not None and mol.symmetry:
        hdiag = hdiag.at[sym_forbid].set(0)
    hdiag = hdiag.ravel()

    mo_coeff = np.asarray(np.hstack((orbo,orbv)))
    vresp = mf.gen_response(singlet=singlet, hermi=0)

    def vind(zs):
        zs = np.asarray(zs).reshape(-1,nocc,nvir)
        if wfnsym is not None and mol.symmetry:
            zs = np.copy(zs)
            zs = zs.at[:,sym_forbid].set(0)

        # *2 for double occupancy
        dmov = np.einsum('xov,qv,po->xpq', zs*2, orbv.conj(), orbo)
        v1ao = vresp(dmov)
        v1ov = np.einsum('xpq,po,qv->xov', v1ao, orbo.conj(), orbv)
        v1ov += np.einsum('xqs,sp->xqp', zs, fvv)
        v1ov -= np.einsum('xpr,sp->xsr', zs, foo)
        if wfnsym is not None and mol.symmetry:
            v1ov = v1ov.at[:,sym_forbid].set(0)
        return v1ov.reshape(v1ov.shape[0],-1)

    return vind, hdiag
gen_tda_hop = gen_tda_operation

def get_ab(mf, mo_energy=None, mo_coeff=None, mo_occ=None):
    r'''A and B matrices for TDDFT response function.

    A[i,a,j,b] = \delta_{ab}\delta_{ij}(E_a - E_i) + (ia||bj)
    B[i,a,j,b] = (ia||jb)
    '''
    if mo_energy is None:
        mo_energy = mf.mo_energy
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    if mo_occ is None:
        mo_occ = mf.mo_occ
    assert mo_coeff.dtype == np.double

    nmo = mo_coeff.shape[-1]
    occidx = np.where(mo_occ==2)[0]
    viridx = np.where(mo_occ==0)[0]
    orbv = mo_coeff[:,viridx]
    orbo = mo_coeff[:,occidx]
    nvir = orbv.shape[1]
    nocc = orbo.shape[1]
    mo = np.hstack((orbo,orbv))
    nmo = nocc + nvir

    e_ia = -mo_energy[occidx][:,None] + mo_energy[viridx][None,:]
    a = np.diag(e_ia.ravel()).reshape(nocc,nvir,nocc,nvir)
    b = np.zeros_like(a)

    def add_hf(a, b, hyb=1):
        eri_mo = ao2mo.incore.general(mf._eri, [orbo,mo,mo,mo], compact=False)
        eri_mo = eri_mo.reshape(nocc,nmo,nmo,nmo)
        a  = np.einsum('iabj->iajb', eri_mo[:nocc,nocc:,nocc:,:nocc]) * 2
        a -= np.einsum('ijba->iajb', eri_mo[:nocc,:nocc,nocc:,nocc:]) * hyb

        b  = np.einsum('iajb->iajb', eri_mo[:nocc,nocc:,:nocc,nocc:]) * 2
        b -= np.einsum('jaib->iajb', eri_mo[:nocc,nocc:,:nocc,nocc:]) * hyb
        return a, b

    a_hf, b_hf = add_hf(a,b)
    a += a_hf
    b += b_hf
    return a, b

def cis_ovlp(mol1, mol2, mo1, mo2, nocc1, nocc2, nmo1, nmo2, x1, x2):
    s_ao = mole.intor_cross('int1e_ovlp', mol1, mol2)
    nvir1 = nmo1 - nocc1

    idx1 = []
    for i in range(nocc1):
        for a in range(nocc1,nmo1):
            idx = numpy.arange(nocc1)
            idx[i] = a
            idx1.append(idx)
    idx1 = np.asarray(idx1)

    if nocc1 == nocc2 and nmo1 == nmo2:
        idx2 = idx1
    else:
        idx2 = []
        for j in range(nocc2):
            for b in range(nocc2,nmo2):
                idx = numpy.arange(nocc2)
                idx[j] = b
                idx2.append(idx)
        idx2 = np.asarray(idx2)

    def body(idx, mo1_occ):
        s_mo = np.einsum('ui,uv,vj->ij', mo1_occ, s_ao, mo2[:,idx])
        return np.linalg.det(s_mo)

    res = 0.
    for i in range(nocc1):
        for a in range(nvir1):
            mo1_occ = mo1[:,idx1[i*nvir1+a]]
            s12 = vmap(body, (0,None))(idx2, mo1_occ)
            res += x1[i,a] * (s12 * x2.ravel()).sum()
    return res

# pylint: disable=abstract-method
@util.pytree_node(Traced_Attributes, num_args=1)
class TDBase(pyscf_tdrhf.TDBase):
    def __init__(self, mf, **kwargs):
        pyscf_tdrhf.TDBase.__init__(self, mf)
        self.__dict__.update(kwargs)

    def get_ab(self, mf=None):
        if mf is None:
            mf = self._scf
        return get_ab(mf)

    def get_precond(self, hdiag):
        def precond(x, e, x0):
            diagd = hdiag - (e-self.level_shift)
            diagd = ops.index_update(diagd, ops.index[abs(diagd)<1e-8], 1e-8)
            return x/diagd
        return precond

@util.pytree_node(Traced_Attributes, num_args=1)
class TDA(TDBase, pyscf_tdrhf.TDA):
    def gen_vind(self, mf=None):
        if mf is None:
            mf = self._scf
        return gen_tda_hop(mf, singlet=self.singlet, wfnsym=self.wfnsym)

    def kernel(self, x0=None, nstates=None):
        cpu0 = (logger.process_clock(), logger.perf_counter())
        self.check_sanity()
        self.dump_flags()
        if nstates is None:
            nstates = self.nstates
        else:
            self.nstates = nstates

        log = logger.Logger(self.stdout, self.verbose)

        vind, hdiag = self.gen_vind(self._scf)
        precond = self.get_precond(hdiag)

        if x0 is None:
            x0 = self.init_guess(stop_grad(self._scf), self.nstates)

        def pickeig(w, v, nroots, envs):
            idx = numpy.where(w > self.positive_eig_threshold)[0]
            return w[idx], v[:,idx], idx
        self.converged, self.e, x1 = davidson1(vind, x0, precond,
                                               tol=self.conv_tol,
                                               nroots=nstates, lindep=self.lindep,
                                               max_cycle=self.max_cycle,
                                               max_space=self.max_space, pick=pickeig,
                                               verbose=log)

        nocc = (self._scf.mo_occ>0).sum()
        nmo = self._scf.mo_occ.size
        nvir = nmo - nocc
# 1/sqrt(2) because self.x is for alpha excitation amplitude and 2(X^+*X) = 1
        self.xy = [(xi.reshape(nocc,nvir)*numpy.sqrt(.5),0) for xi in x1]

        if self.chkfile:
            chkfile.save(self.chkfile, 'tddft/e', self.e)
            chkfile.save(self.chkfile, 'tddft/xy', self.xy)

        log.timer('TDA', *cpu0)
        self._finalize()
        return self.e, self.xy

CIS = TDA
