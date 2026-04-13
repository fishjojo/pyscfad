# Copyright 2023-2026 The PySCFAD Authors
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
import jax
import numpy
from jax import numpy as np
from pyscfad import config
from pyscfad.ao2mo import _ao2mo
from pyscfad.mp.mp2 import _gamma1_intermediates
from pyscfad.scf import cphf
from pyscfad.implicit_diff import make_implicit_diff
from pyscfad.scipy.sparse.linalg import gmres
from pyscfad.tools.linear_solver import GMRESDisp

def fock_response_rhf(mf, dm, mo_coeff=None, mo_occ=None, full=True, einsum=np.einsum):
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    if mo_occ is None:
        mo_occ = mf.mo_occ
    Ci = mo_coeff[:, mo_occ>0]
    Ca = mo_coeff[:, mo_occ==0]
    if full:
        dmao = einsum('xp,pq,yq->xy', mo_coeff, dm, mo_coeff)
    else:
        dmao = einsum('xa,ai,yi->xy', Ca, dm, Ci)
    rao = 2.0 * mf.get_veff(dm=dmao+dmao.T)
    rvo = einsum('xa,xy,yi->ai', Ca, rao, Ci)
    return rvo

def make_rdm1_vo_frag(mp, dm1_oo, dm1_vv, Lia, Ljb, eia, ejb, eris=None, ao_repr=False):
    mf = mp._scf
    mo_occ = mf.mo_occ
    orbo, orbv = mp.split_mo()[1:3]
    moe_occ, moe_vir = mp.split_moe()[1:3]
    mo_coeff = np.concatenate((orbo, orbv), axis=-1)
    mo_energy = np.concatenate((moe_occ, moe_vir))
    # TODO also consider frozen orbitals
    assert len(mo_occ) == len(mo_energy)

    if eris is None:
        eris = mp.ao2mo()

    nocc = orbo.shape[-1]
    nmo = mo_coeff.shape[-1]
    nvir = nmo - nocc

    eri1 = mf.with_df._cderi
    Loo = _ao2mo.nr_e2(eri1, mo_coeff, (0, nocc, 0, nocc), aosym='s2')
    Lvv = _ao2mo.nr_e2(eri1, mo_coeff, (nocc, nmo, nocc, nmo), aosym='s2')
    Loo = Loo.reshape(-1, nocc, nocc)
    Lvv = Lvv.reshape(-1, nvir, nvir)

    # make t2
    @jax.checkpoint
    def _fn(dummy, x):
        La, ea = x
        gi = np.dot(La.T, Ljb.reshape(-1,nocc*nvir)).reshape((nvir,nocc,nvir))
        gi = gi.transpose(1,0,2)
        t2i = gi / (ejb[:,None,:] + ea[None,:,None])
        return None, t2i
    _, t2 = jax.lax.scan(_fn, None, (Lia.transpose(1,0,2), eia))

    dm1 = np.zeros((nmo, nmo))
    dm1 = dm1.at[:nocc,:nocc].set(dm1_oo)
    dm1 = dm1.at[nocc:,nocc:].set(dm1_vv)
    Rvo = -fock_response_rhf(mf, dm1, mo_coeff=mo_coeff, mo_occ=mo_occ, full=True)

    tt = 2 * t2 - t2.transpose(0,1,3,2)
    tmp = np.einsum('jkab,Lkb->Lja', tt.transpose(1,0,2,3), Lia)
    Rvo += np.einsum('Lja,Lji->ai', tmp, Loo) * 2
    Rvo -= np.einsum('Lib,Lab->ai', tmp, Lvv) * 2

    if config.moleintor_opt:
        zvo = None
        disp = GMRESDisp(disp=mf.mol.verbose>4)
        cphf_solver = make_implicit_diff(solve_cphf, True,
                        fixed_point=False,
                        optimality_cond=cphf_optcond,
                        solver=partial(gmres, tol=1e-9, maxiter=50,
                                       callback=disp, callback_type='pr_norm'),
                        has_aux=False)
        zvo = cphf_solver(zvo, mf, -Rvo, mo_energy, mo_coeff)
    else:
        def fvind(z):
            return fock_response_rhf(mf, z.reshape(Rvo.shape), mo_coeff=mo_coeff,
                                     mo_occ=mo_occ, full=False)
        zvo = cphf.solve(fvind, mo_energy, mo_occ, -Rvo)[0]

    if ao_repr:
        Ci = mo_coeff[:, mo_occ>0]
        Ca = mo_coeff[:, mo_occ==0]
        out = np.einsum('ua,ai,vi->uv', Ca, zvo, Ci)
        out += out.T
    else:
        out = zvo
    return out

def cphf_optcond(z, mf, h1, mo_energy, mo_coeff):
    mo_occ = mf.mo_occ
    e_a = mo_energy[mo_occ==0]
    e_i = mo_energy[mo_occ>0]
    e_ai = 1/(e_a[:,None] - e_i[None,:])

    def fvind(z):
        return fock_response_rhf(mf, z.reshape(h1.shape), mo_coeff=mo_coeff,
                                 mo_occ=mo_occ, full=False)

    def vind_vo(mo1):
        v = fvind(mo1.reshape(h1.shape)).reshape(h1.shape)
        v *= e_ai
        return v

    mo1base = h1 * (-e_ai)
    zero = z.reshape(h1.shape) + vind_vo(z) - mo1base
    return zero

def solve_cphf(z, mf, h1, mo_energy, mo_coeff, *,
               tol=1e-9, max_cycle=50):
    from pyscf.scf import cphf as pyscf_cphf
    mo_occ = mf.mo_occ
    e_a = mo_energy[mo_occ==0]
    e_i = mo_energy[mo_occ>0]
    e_ai = e_a[:,None] - e_i[None,:]

    def fvind(z):
        return fock_response_rhf(mf, z.reshape(h1.shape), mo_coeff=mo_coeff,
                                 mo_occ=mo_occ, full=False, einsum=numpy.einsum)

    def vind_vo(mo1):
        v  = fvind(mo1.reshape(h1.shape)).reshape(h1.shape)
        v += e_ai * mo1.reshape(h1.shape)
        return -v.ravel()

    mo1 = pyscf_cphf.solve(fvind, mo_energy, mo_occ, h1,
                           max_cycle=max_cycle, tol=tol,
                           verbose=mf.mol.verbose)[0]
    return mo1.reshape(h1.shape)

def make_rdm1_vo(mp, eris=None, ao_repr=False):
    mf = mp._scf
    mo_occ = mf.mo_occ
    orbo, orbv = mp.split_mo()[1:3]
    moe_occ, moe_vir = mp.split_moe()[1:3]
    mo_coeff = np.concatenate((orbo, orbv), axis=-1)
    mo_energy = np.concatenate((moe_occ, moe_vir))
    # TODO also consider frozen orbitals
    assert len(mo_occ) == len(mo_energy)

    if eris is None:
        eris = mp.ao2mo()

    nocc = orbo.shape[-1]
    nmo = mo_coeff.shape[-1]
    nvir = nmo - nocc

    Lov = eris.Lov
    eri1 = mf.with_df._cderi
    Loo = _ao2mo.nr_e2(eri1, mo_coeff, (0, nocc, 0, nocc), aosym='s2')
    Lvv = _ao2mo.nr_e2(eri1, mo_coeff, (nocc, nmo, nocc, nmo), aosym='s2')
    Loo = Loo.reshape(-1, nocc, nocc)
    Lvv = Lvv.reshape(-1, nvir, nvir)

    # make t2
    eia = mo_energy[:nocc,None] - mo_energy[None,nocc:]

    @jax.checkpoint
    def _fn(dummy, x):
        La, ea = x
        gi = np.dot(La.T, Lov.reshape(-1,nocc*nvir)).reshape((nvir,nocc,nvir))
        gi = gi.transpose(1,0,2)
        t2i = gi / (eia[:,:,None] + ea[None,None,:])
        return None, t2i
    _, t2 = jax.lax.scan(_fn, None, (Lov.transpose(1,0,2), eia))

    dm1_oo, dm1_vv = _gamma1_intermediates(mp, t2=t2)

    dm1 = np.zeros((nmo, nmo))
    dm1 = dm1.at[:nocc,:nocc].set(dm1_oo)
    dm1 = dm1.at[nocc:,nocc:].set(dm1_vv)
    Rvo = -fock_response_rhf(mf, dm1, mo_coeff=mo_coeff, mo_occ=mo_occ, full=True)

    tt = 2 * t2 - t2.transpose(0,1,3,2)
    tmp = np.einsum('jkab,Lkb->Lja', tt, Lov)
    Rvo += np.einsum('Lja,Lji->ai', tmp, Loo) * 2
    Rvo -= np.einsum('Lib,Lab->ai', tmp, Lvv) * 2

    if config.moleintor_opt:
        zvo = None
        disp = GMRESDisp(disp=mf.mol.verbose>4)
        cphf_solver = make_implicit_diff(solve_cphf, True,
                        fixed_point=False,
                        optimality_cond=cphf_optcond,
                        solver=partial(gmres, tol=1e-9, maxiter=50,
                                       callback=disp, callback_type='pr_norm'),
                        has_aux=False)
        zvo = cphf_solver(zvo, mf, -Rvo, mo_energy, mo_coeff)
    else:
        def fvind(z):
            return fock_response_rhf(mf, z.reshape(Rvo.shape), mo_coeff=mo_coeff,
                                     mo_occ=mo_occ, full=False)
        zvo = cphf.solve(fvind, mo_energy, mo_occ, -Rvo)[0]

    if ao_repr:
        Ci = mo_coeff[:, mo_occ>0]
        Ca = mo_coeff[:, mo_occ==0]
        out = np.einsum('ua,ai,vi->uv', Ca, zvo, Ci)
        out += out.T
    else:
        out = zvo
    return out
