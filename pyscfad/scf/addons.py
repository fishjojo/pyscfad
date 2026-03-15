# Copyright 2021-2026 The PySCFAD Authors
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

from functools import partial
from jax import custom_jvp
from jax.lax import while_loop
from pyscfad import numpy as np
from pyscfad import scipy

def _fermi_entropy(mo_occ, occ_thresh=1e-10):
    occ = mo_occ / 2.0
    thr = occ_thresh
    occ_safe = np.where(np.logical_or(occ < thr, occ > 1 - thr), 0.5, occ)
    ent_term = occ_safe * np.log(occ_safe) + (1 - occ_safe) * np.log(1 - occ_safe)

    return -2 * np.where(
        np.logical_or(occ < thr, occ > 1 - thr),
        0.,
        ent_term
    ).sum()

def _fermi_smearing_occ(mu, mo_energy, sigma, mo_mask):
    de = (mo_energy - mu) / sigma
    de = np.where(np.less(de, 40.), de, np.inf)
    occ = 1. / (np.exp(de) + 1.)
    occ = np.where(mo_mask, occ, 0.)
    return occ

@partial(custom_jvp, nondiff_argnums=(0, 3))
def _smearing_solve_mu(f_occ, mo_es, nocc, sigma, mo_mask):
    def cond_fun(value):
        _, nerr = value
        return abs(nerr) > 1e-8

    def body_fun(value):
        """One Halley step"""
        mu, nerr = value
        occ = f_occ(mu, mo_es, sigma, mo_mask)
        grad = occ * (1.-occ) / sigma
        hess = grad * (1.-2*occ) / sigma
        nerr = np.sum(occ) - nocc
        grad = np.sum(grad)
        hess = np.sum(hess)
        dmu = -nerr * grad / (grad**2 - .5 * hess * nerr)
        return mu + dmu, nerr

    mu, _ = while_loop(cond_fun, body_fun, (mo_es[nocc-1], 1e2))
    return mu

@_smearing_solve_mu.defjvp
def _smearing_solve_mu_jvp(f_occ, sigma, primals, tangents):
    mo_es, nocc, mo_mask, = primals
    dmo_e, *_, = tangents

    mu = _smearing_solve_mu(f_occ, mo_es, nocc, sigma, mo_mask)
    occ = f_occ(mu, mo_es, sigma, mo_mask)
    dndmu = occ * (1.-occ) / sigma
    return mu, np.dot(dndmu, dmo_e) / np.sum(dndmu)

def _smearing_optimize(f_occ, mo_es, nocc, sigma, mo_mask):
    mu = _smearing_solve_mu(f_occ, mo_es, nocc, sigma, mo_mask)
    mo_occ = f_occ(mu, mo_es, sigma, mo_mask)
    return mu, mo_occ

def get_occ_smearing(mo_energy, nocc, sigma, mo_mask, method="fermi"):
    """Get MO occupations with smearing.
    """
    if method.lower() == "fermi":
        f_occ = _fermi_smearing_occ
    else:
        raise NotImplementedError

    _, mo_occ = _smearing_optimize(f_occ, mo_energy, nocc, sigma, mo_mask)
    return mo_occ

def canonical_orth_(S, thr=1e-7):
    """Löwdin's canonical orthogonalization.
    """
    # Ensure the basis functions are normalized (symmetry-adapted ones are not!)
    normlz = np.power(np.diag(S), -0.5)
    Snorm = normlz[:,None] * S * normlz[None,:]
    # Form vectors for normalized overlap matrix
    Sval, Svec = scipy.linalg.eigh(Snorm)
    X = Svec[:,Sval>=thr] / np.sqrt(Sval[Sval>=thr])
    # Plug normalization back in
    X = normlz[:,None] * X
    return X
