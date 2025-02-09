from functools import wraps
from pyscf.scf import addons as pyscf_addons
from pyscfad import numpy as np
from pyscfad import scipy

@wraps(pyscf_addons.canonical_orth_)
def canonical_orth_(S, thr=1e-7):
    # Ensure the basis functions are normalized (symmetry-adapted ones are not!)
    normlz = np.power(np.diag(S), -0.5)
    Snorm = np.dot(np.diag(normlz), np.dot(S, np.diag(normlz)))
    # Form vectors for normalized overlap matrix
    Sval, Svec = scipy.linalg.eigh(Snorm)
    X = Svec[:,Sval>=thr] / np.sqrt(Sval[Sval>=thr])
    # Plug normalization back in
    X = np.dot(np.diag(normlz), X)
    return X
