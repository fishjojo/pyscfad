import numpy
from pyscf.scf import diis as pyscf_cdiis
from pyscfad import numpy as np
from pyscfad import ops
from pyscfad import lib
from pyscfad.lib import logger

class CDIIS(lib.diis.DIIS, pyscf_cdiis.CDIIS):
    def __init__(self, mf=None, filename=None, Corth=None):
        pyscf_cdiis.CDIIS.__init__(self, mf=mf, filename=filename, Corth=Corth)
        self.incore = True

    def update(self, s, d, f, *args, **kwargs):
        errvec = get_err_vec(s, d, f, self.Corth)
        # no need to trace error vectors
        errvec = ops.to_numpy(errvec)
        logger.debug1(self, 'diis-norm(errvec)=%g', numpy.linalg.norm(errvec))
        f_prev = kwargs.get('f_prev', None)
        if abs(self.damp) < 1e-6 or f_prev is None:
            xnew = lib.diis.DIIS.update(self, f, xerr=errvec)
        else:
            f = f*(1-self.damp) + f_prev*self.damp
            xnew = lib.diis.DIIS.update(self, f, xerr=errvec)
        if self.rollback > 0 and len(self._bookkeep) == self.space:
            self._bookkeep = self._bookkeep[-self.rollback:]
        return xnew

@ops.jit
def get_err_vec_orig(s, d, f):
    if f.ndim == 2:
        sdf = s @ d @ f
        errvec = (sdf.conj().T - sdf).ravel()

    elif f.ndim == 3 and s.ndim == 3:
        errvec = []
        for i in range(f.shape[0]):
            sdf = s[i] @ d[i] @ f[i]
            errvec.append((sdf.conj().T - sdf).ravel())
        errvec = np.hstack(errvec)

    elif f.ndim == s.ndim+1 and f.shape[0] == 2:  # for UHF
        errvec = np.hstack([
            get_err_vec_orig(s, d[0], f[0]).ravel(),
            get_err_vec_orig(s, d[1], f[1]).ravel()])
    else:
        raise RuntimeError('Unknown SCF DIIS type')
    return errvec

@ops.jit
def get_err_vec_orth(s, d, f, Corth):
    if f.ndim == 2:
        sdf = Corth.conj().T @ s @ d @ f @ Corth
        errvec = (sdf.conj().T - sdf).ravel()

    elif f.ndim == 3 and s.ndim == 3:
        errvec = []
        for i in range(f.shape[0]):
            sdf = Corth[i].conj().T @ s[i] @ d[i] @ f[i] @ Corth[i]
            errvec.append((sdf.conj().T - sdf).ravel())
        errvec = np.hstack(errvec)

    elif f.ndim == s.ndim+1 and f.shape[0] == 2:  # for UHF
        errvec = np.hstack([
            get_err_vec_orth(s, d[0], f[0], Corth[0]).ravel(),
            get_err_vec_orth(s, d[1], f[1], Corth[1]).ravel()])
    else:
        raise RuntimeError('Unknown SCF DIIS type')
    return errvec

def get_err_vec(s, d, f, Corth=None):
    if Corth is None:
        return get_err_vec_orig(s, d, f)
    else:
        return get_err_vec_orth(s, d, f, Corth)

SCFDIIS = SCF_DIIS = DIIS = CDIIS
