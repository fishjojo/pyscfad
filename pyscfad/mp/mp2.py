from pyscf.lib import logger, split_reshape
from pyscf.mp import mp2 as pyscf_mp2
from pyscfad import util
from pyscfad import lib
from pyscfad.lib import numpy as np

# Iteratively solve MP2 if non-canonical HF is provided
def _iterative_kernel(mp, eris, verbose=None):
    cput1 = cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.new_logger(mp, verbose)

    emp2, t2 = mp.init_amps(eris=eris)
    log.info('Init E(MP2) = %.15g', emp2)

    adiis = lib.diis.DIIS(mp)

    conv = False
    for istep in range(mp.max_cycle):
        t2new = mp.update_amps(t2, eris)

        if hasattr(t2new, 'ndim'):
            normt = np.linalg.norm(t2new - t2)
            t2 = None
            t2new = adiis.update(t2new)
        else: # UMP2
            normt = np.linalg.norm([np.linalg.norm(t2new[i] - t2[i])
                                     for i in range(3)])
            t2 = None
            t2shape = [x.shape for x in t2new]
            t2new = np.hstack([x.ravel() for x in t2new])
            t2new = adiis.update(t2new)
            t2new = split_reshape(t2new, t2shape)

        t2, t2new = t2new, None
        emp2, e_last = mp.energy(t2, eris), emp2
        log.info('cycle = %d  E_corr(MP2) = %.15g  dE = %.9g  norm(t2) = %.6g',
                 istep+1, emp2, emp2 - e_last, normt)
        cput1 = log.timer('MP2 iter', *cput1)
        if abs(emp2-e_last) < mp.conv_tol and normt < mp.conv_tol_normt:
            conv = True
            break
    log.timer('MP2', *cput0)
    del log
    return conv, emp2, t2

@util.pytree_node(['_scf', 'mol'], num_args=1)
class MP2(pyscf_mp2.MP2):
    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None, **kwargs):
        pyscf_mp2.MP2.__init__(self, mf, frozen=frozen,
                               mo_coeff=mo_coeff, mo_occ=mo_occ)
        self.__dict__.update(kwargs)

    def ao2mo(self, mo_coeff=None):
        eris = pyscf_mp2._ChemistsERIs()
        eris._common_init_(self, mo_coeff)
        mo_coeff = eris.mo_coeff

        nocc = self.nocc
        co = np.asarray(mo_coeff[:,:nocc])
        cv = np.asarray(mo_coeff[:,nocc:])
        eris.ovov = np.einsum("uvst,ui,va,sj,tb->iajb", self._scf._eri, co,cv,co,cv)
        return eris

    _iterative_kernel = _iterative_kernel
