from functools import partial
import numpy
import jax
from jax import custom_vjp
from pyscf import __config__
from pyscf.lib import direct_sum, current_memory
#from pyscf.mp.mp2 import _ChemistsERIs
from pyscfad import config
from pyscfad import numpy as np
from pyscfad import util
from pyscfad.ops import vmap
from pyscfad.lib import logger
from pyscfad.ao2mo import _ao2mo
from pyscfad.mp import mp2

WITH_T2 = getattr(__config__, 'mp_dfmp2_with_t2', True)

def _contract(Lov, mo_energy, nocc, nvir, with_t2=True):
    def body(Lv, Lov, ea, eia):
        gi = np.einsum('la,ljb->jab', Lv, Lov)
        t2i = gi / (eia[:,:,None] + ea[None,None,:])
        ei = np.einsum('jab,jab', t2i, gi) * 2 - np.einsum('jab,jba', t2i, gi)
        return ei, t2i

    Lov = Lov.reshape((-1, nocc, nvir))
    eia = mo_energy[:nocc,None] - mo_energy[None,nocc:]
    e, t2 = vmap(body, in_axes=(1,None,0,None),
                 signature='(l,a),(b)->(),(j,a,b)')(Lov, Lov, eia, eia)
    if not with_t2:
        t2 = None
    emp2 = e.sum().real
    return emp2, t2

@partial(custom_vjp, nondiff_argnums=(2,3,4))
def _contract_opt(Lov, mo_energy, nocc, nvir, with_t2=True):
    if with_t2:
        t2 = numpy.empty((nocc,nocc,nvir,nvir))
    else:
        t2 = None
    eia = mo_energy[:nocc,None] - mo_energy[None,nocc:]

    emp2 = 0
    for i in range(nocc):
        buf = numpy.dot(Lov[:,i*nvir:(i+1)*nvir].T,
                        Lov).reshape((nvir,nocc,nvir))
        gi = buf.transpose(1,0,2)
        t2i = gi / direct_sum('jb+a->jba', eia, eia[i])
        emp2 += numpy.einsum('jab,jab', t2i, gi) * 2
        emp2 -= numpy.einsum('jab,jba', t2i, gi)
        if with_t2:
            t2[i] = t2i
    return emp2, t2

def _contract_opt_fwd(Lov, mo_energy, nocc, nvir, with_t2):
    emp2, t2 = _contract_opt(Lov, mo_energy, nocc, nvir, with_t2)
    return (emp2, t2), (Lov, mo_energy, t2)

def _contract_opt_bwd(nocc, nvir, with_t2,
                      res, ybar):
    Lov, mo_energy, t2 = res
    emp2_bar = ybar[0]
    if with_t2:
        t2 = numpy.asarray(t2)
        t2_bar = numpy.asarray(ybar[1])

    eia = mo_energy[:nocc,None] - mo_energy[None,nocc:]
    Lov_bar = numpy.zeros_like(Lov)
    mo_energy_bar = numpy.zeros_like(mo_energy)
    for i in range(nocc):
        ejab = direct_sum('jb+a->jba', eia, eia[i])
        if with_t2:
            t2i = t2[i]
            gi = t2i * ejab
        else:
            gi = numpy.dot(Lov[:,i*nvir:(i+1)*nvir].T,
                           Lov).reshape((nvir,nocc,nvir))
            gi = gi.transpose(1,0,2)
            t2i = gi / ejab

        buf = emp2_bar * t2i
        gi_bar = 2 * buf - buf.transpose(0,2,1)
        buf = emp2_bar * gi
        t2i_bar = 2 * buf - buf.transpose(0,2,1)
        if with_t2:
            t2i_bar += t2_bar[i]
        gi_bar += t2i_bar / ejab

        buf = gi_bar.transpose(1,0,2).reshape(nvir,-1)
        Lov_bar += numpy.dot(Lov[:,i*nvir:(i+1)*nvir], buf)
        Lov_bar[:,i*nvir:(i+1)*nvir] += numpy.dot(Lov, buf.T)
        gi_bar = None

        buf = -gi * t2i_bar / (ejab * ejab)
        ejab = gi = t2i_bar = None
        mo_energy_bar[i] += numpy.sum(buf)
        mo_energy_bar[:nocc] += numpy.sum(buf.reshape(nocc,-1), axis=-1)
        mo_energy_bar[nocc:] -= numpy.sum(buf.reshape(-1,nvir), axis=0)
        mo_energy_bar[nocc:] -= numpy.sum(buf.transpose(1,0,2).reshape(nvir,-1), axis=-1)
        buf = None
    return (Lov_bar, mo_energy_bar)

_contract_opt.defvjp(_contract_opt_fwd, _contract_opt_bwd)

def _contract_scan(Lov, mo_energy, nocc, nvir, with_t2=True):
    if with_t2:
        t2 = numpy.empty((nocc,nocc,nvir,nvir))
    else:
        t2 = None
    eia = mo_energy[:nocc,None] - mo_energy[None,nocc:]

    @jax.checkpoint
    def _fn(emp2, x):
        La, ea = x
        gi = np.dot(La.T, Lov).reshape((nvir,nocc,nvir))
        gi = gi.transpose(1,0,2)
        t2i = gi / (eia[:,:,None] + ea[None,None,:])
        emp2 += np.einsum('jab,jab', t2i, gi) * 2
        emp2 -= np.einsum('jab,jba', t2i, gi)
        if not with_t2:
            t2i = None
        return emp2, t2i

    emp2, t2 = jax.lax.scan(_fn, 0., (Lov.reshape((-1,nocc,nvir)).transpose(1,0,2), eia))
    if not with_t2:
        t2 = None
    return emp2, t2

def kernel(mp, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2,
           verbose=None):
    if mo_energy is not None or mo_coeff is not None:
        assert (mp.frozen == 0 or mp.frozen is None)

    if eris is None:
        eris = mp.ao2mo(mo_coeff)
    if mo_energy is None:
        mo_energy = eris.mo_energy
    if mo_coeff is None:
        mo_coeff = eris.mo_coeff

    nocc = mp.nocc
    nvir = mp.nmo - nocc
    #naux = mp.with_df.get_naoaux()

    Lov = mp.loop_ao2mo(mo_coeff, nocc, with_t2)

    if config.moleintor_opt:
        #emp2, t2 = _contract_opt(Lov, mo_energy, nocc, nvir, with_t2)
        emp2, t2 = _contract_scan(Lov, mo_energy, nocc, nvir, with_t2)
    else:
        emp2, t2 = _contract(Lov, mo_energy, nocc, nvir, with_t2)
    return emp2, t2

@util.pytree_node(['_scf', 'mol', 'with_df'], num_args=1)
class MP2(mp2.MP2):
    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None, **kwargs):
        super().__init__(mf, frozen=frozen,
                         mo_coeff=mo_coeff, mo_occ=mo_occ, **kwargs)
        if getattr(mf, 'with_df', None):
            self.with_df = mf.with_df
        else:
            raise KeyError('The mean-field object has no density fitting.')

        self._keys.update(['with_df'])
        self.__dict__.update(kwargs)

    def ao2mo(self, mo_coeff=None):
        eris = mp2._ChemistsERIs()
        eris._common_init_(self, mo_coeff)
        return eris

    def loop_ao2mo(self, mo_coeff, nocc, with_t2=WITH_T2):
        # NOTE return the whole 3c integral for now
        nao, nmo = mo_coeff.shape
        nvir = nmo - nocc
        ijslice = (0, nocc, nocc, nmo)

        with_df = self.with_df
        naux = with_df.get_naoaux()
        mem_incore = (naux*nocc*nvir + nocc*nvir*nvir*2) * 8 / 1e6
        if with_t2:
            mem_incore += (nocc*nvir)**2 * 8 / 1e6
        mem_now = current_memory()[0]
        if (mem_incore + mem_now < self.max_memory) or self.mol.incore_anyway:
            eri1 = with_df._cderi
            Lov = _ao2mo.nr_e2(eri1, mo_coeff, ijslice, aosym='s2')
            return Lov
        else:
            raise RuntimeError(f'{mem_incore+mem_now} MB of memory is needed.')

    def kernel(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2):
        if self.verbose >= logger.WARN:
            self.check_sanity()

        self.dump_flags()

        self.e_hf = self.get_e_hf(mo_coeff=mo_coeff)

        if eris is None:
            eris = self.ao2mo(mo_coeff)

        if self._scf.converged:
            self.e_corr, self.t2 = self.init_amps(mo_energy, mo_coeff, eris, with_t2)
        else:
            raise NotImplementedError

        # TODO SCS-MP2
        self.e_corr_ss = 0
        self.e_corr_os = 0
        self._finalize()
        return self.e_corr, self.t2

    def init_amps(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2):
        return kernel(self, mo_energy, mo_coeff, eris, with_t2)

del WITH_T2
