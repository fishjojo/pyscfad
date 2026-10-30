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

"""
Density-fitted restricted CCSD.
"""
from functools import partial
import jax
from jax.custom_derivatives import SymbolicZero

from pyscf.lib import current_memory
from pyscfad import numpy as np
from pyscfad import ops
from pyscfad import lib
from pyscfad import config
from pyscfad.lib import logger
from pyscfad.ao2mo import _ao2mo
from pyscfad.cc import ccsd

@jax.jit
def _build_eris_vvvv(Lvv):
    #: (ab|cd) + (cb|ad) and (ab|cd) - (cb|ad) for a >= c, d >= b
    naux, nvir, _ = Lvv.shape
    npair = nvir * (nvir + 1) // 2
    # GEMM operands are materialized in the layout the GEMM reads (XLA's CPU
    # dot is ~2x slower on a transposed operand); optimization_barrier keeps
    # XLA from folding the transpose back into the dot
    Lv2T = jax.lax.optimization_barrier(Lvv.reshape(naux, nvir * nvir).T)

    def body(V, a):
        # X[c,d,b] = (ab|cd)
        X = (Lv2T @ Lvv[:, a, :]).reshape(nvir, nvir, nvir)
        Xt = X.transpose(0, 2, 1)
        W = np.stack((lib.pack_tril(X + Xt), lib.pack_tril(X - Xt)))
        # rows (a,c) of c > a are overwritten by the next a
        return jax.lax.dynamic_update_slice(V, W, (0, a * (a + 1) // 2, 0)), None

    V = np.zeros((2, npair, npair), dtype=Lvv.dtype)
    return jax.lax.scan(body, V, np.arange(nvir))[0]

def _G_dense(tau, V):
    # the parts of tau symmetric and antisymmetric in the virtual indices,
    # over a >= c; the diagonal is counted twice in V[0]
    nvir = tau.shape[-1]
    taut = tau.transpose(0, 2, 1)
    tau_s = lib.pack_tril(tau + taut) * (.5 - .25 * lib.pack_tril(np.eye(nvir)))
    tau_a = lib.pack_tril(tau - taut) * .5
    # V[0] and V[1] are symmetric; one batched dot_general reads V as stored
    Gs, Ga = jax.lax.dot_general(np.stack((tau_s, tau_a)), V, (((2,), (2,)), ((0,), (0,))))
    return (lib.unpack_tril(Gs, filltriu=lib.SYMMETRIC)
            + lib.unpack_tril(Ga, filltriu=lib.ANTIHERMI))

def _G_direct(tau, L):
    o2, nvir, _ = tau.shape
    v2 = nvir * nvir
    naux = L.shape[0]
    L2 = L.reshape(naux, v2)
    tau2 = tau.reshape(o2, v2)
    LT = jax.lax.optimization_barrier(L.transpose(2, 1, 0))

    @jax.checkpoint
    def slab(LTx):
        Vx = (LTx @ L2).reshape(v2, nvir)
        return tau2 @ Vx

    out = jax.lax.map(slab, LT)
    return out.transpose(1, 2, 0)

def _G_direct_acc(tau, L):
    o2, nvir, _ = tau.shape
    v2 = nvir * nvir
    naux = L.shape[0]
    L2 = L.reshape(naux, v2)
    LT = jax.lax.optimization_barrier(L.transpose(2, 1, 0))
    tauT = jax.lax.optimization_barrier(tau.transpose(2, 0, 1))

    # as _G_direct, accumulated over the slabs: G[p,(y,z)] = sum_{x,w}
    # tau[p,w,x] (yx|zw). Contracting the short axis of a slab is faster
    # than contracting its long axis as _G_direct does.
    @jax.checkpoint
    def slab(G, xs):
        LTx, tau_x = xs
        Vx = (LTx @ L2).reshape(v2, nvir)
        return G + tau_x @ Vx.T, None

    G = np.zeros((o2, v2), dtype=tau.dtype)
    G = jax.lax.scan(slab, G, (LT, tauT))[0]
    return G.reshape(o2, nvir, nvir)

@partial(jax.jit, static_argnames=('tangent',))
def _G(tau, L, V, tangent=False):
    if V is None:
        # reverse mode transposes the slabs of _G_direct into those of
        # _G_direct_acc, so the tangent of tau is computed by _G_direct
        if tangent:
            return _G_direct(tau, L)
        return _G_direct_acc(tau, L)
    return _G_dense(tau, V)

@jax.jit
@jax.checkpoint
def _G_Ldot(tau, L, L_dot):
    '''Tangent of ``G`` for a tangent ``L_dot`` of the full DF factor,
    ``sum_{z,w} tau[p,z,w] (L_dot[P,z,x] L[P,w,y] + L[P,z,x] L_dot[P,w,y])``.

    Accumulated over slabs ``T_z[w,(y,x)] = sum_P L[P,w,y] L_dot[P,z,x]``;
    the second term is the first one with tau and the result transposed in
    the virtual indices, so ``B`` stacks tau with its transpose. The transpose
    of this loop contracts tau with the cotangent of G one slab at a time.
    '''
    o2, nvir, _ = tau.shape
    naux = L.shape[0]
    L2T = jax.lax.optimization_barrier(L.reshape(naux, nvir * nvir).T) # [(w,y), P]
    B = np.concatenate((tau, tau.transpose(0, 2, 1)), axis=0)
    BT = jax.lax.optimization_barrier(B.transpose(1, 0, 2)) # BT[z] = B[:,z,:]

    def slab(H, xs):
        Bz, L_dot_z = xs
        Tz = (L2T @ L_dot_z).reshape(nvir, nvir * nvir) # [w, (y,x)]
        return H + Bz @ Tz, None

    H0 = np.zeros((2 * o2, nvir * nvir), dtype=tau.dtype)
    H = jax.lax.scan(slab, H0, (BT, np.moveaxis(L_dot, 1, 0)))[0]
    H = H.reshape(2 * o2, nvir, nvir)
    return H[:o2] + H[o2:].transpose(0, 2, 1)

@jax.jit
def _G_Lbar(tau, L, G_bar):
    '''Cotangent of the full DF factor for a cotangent ``G_bar`` of ``G``,
    ``sum_{p,c,d} B[p,x,c] L[P,c,d] C[p,d,y]``, where ``B`` and ``C`` stack tau
    and G_bar with their transposes (L enters ``G`` twice). Unlike the
    transpose of :func:`_G_Ldot`, both GEMMs of a slab read their operands
    as stored.
    '''
    o2, nvir, _ = tau.shape
    naux = L.shape[0]
    B = np.concatenate((tau, tau.transpose(0, 2, 1)), axis=0)
    C = np.concatenate((G_bar, G_bar.transpose(0, 2, 1)), axis=0).reshape(2 * o2, -1)
    BT = jax.lax.optimization_barrier(B.transpose(1, 2, 0)) # BT[x] = B[:,x,:].T
    L2 = L.reshape(naux, -1)

    def slab(Bx):
        Yx = (Bx @ C).reshape(nvir * nvir, nvir) # [(c,d), y]
        return L2 @ Yx

    return jax.lax.map(slab, BT).transpose(1, 0, 2)

@jax.custom_jvp
def ladder(tau, L, V):
    #: G[p,d,b] = sum_{a,c} tau[p,a,c] (ab|cd)``.
    return _G(tau, L, V)

@partial(ladder.defjvp, symbolic_zeros=True)
def _ladder_jvp(primals, tangents):
    tau, L, V = primals
    tau_dot, L_dot, V_dot = tangents
    if not (V_dot is None or isinstance(V_dot, SymbolicZero)):
        raise NotImplementedError('the dense cache vvvv is not differentiated; '
                                  'build it from stop_gradient(Lvv)')
    G_dot = []
    if not isinstance(tau_dot, SymbolicZero):
        G_dot.append(_G(tau_dot, L, V, tangent=True))
    if not isinstance(L_dot, SymbolicZero):
        G_dot.append(_G_Ldot(tau, L, L_dot))
    return _G(tau, L, V), sum(G_dot[1:], G_dot[0])

@jax.custom_vjp
def ladder_rev(tau, L, V):
    '''Reverse-mode only :func:`ladder`.'''
    return _G(tau, L, V)

def _ladder_rev_fwd(tau, L, V):
    # NOTE with symbolic_zeros, the inputs are CustomVJPPrimal, telling
    # _ladder_rev_bwd which cotangents are needed
    perturbed = (tau.perturbed, L.perturbed)
    tau, L = tau.value, L.value
    if V is not None:
        V = V.value
    return _G(tau, L, V), (tau, L, V, perturbed)

def _ladder_rev_bwd(res, G_bar):
    tau, L, V, (tau_perturbed, L_perturbed) = res
    tau_bar = L_bar = None
    if isinstance(G_bar, SymbolicZero):
        return tau_bar, L_bar, None
    # G is a self-adjoint map of tau: its cotangent is the same contraction
    if tau_perturbed:
        tau_bar = _G(G_bar, L, V)
    if L_perturbed:
        L_bar = _G_Lbar(tau, L, G_bar)
    return tau_bar, L_bar, None

ladder_rev.defvjp(_ladder_rev_fwd, _ladder_rev_bwd, symbolic_zeros=True)

def _contract_vvvv_t2(mycc, mol, Lvv, t2, out=None, verbose=None, vvvv=None):
    '''Ht2 = numpy.einsum('ijcd,acbd->ijab', t2, vvvv)
    '''
    nvir = t2.shape[-1]
    Lvv = lib.unpack_tril(Lvv, filltriu=lib.SYMMETRIC)
    fn = ladder_rev if config.moleintor_opt else ladder
    Ht2 = fn(t2.reshape(-1, nvir, nvir), Lvv, vvvv).transpose(0, 2, 1)
    return Ht2.reshape(t2.shape)

class RCCSD(ccsd.CCSD):
    '''Density-fitted RCCSD.
    '''
    _dynamic_attr = ['with_df',]
    _keys = {'with_df', 'vvvv_direct',}

    vvvv_direct = True

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        super().__init__(mf, frozen=frozen, mo_coeff=mo_coeff, mo_occ=mo_occ)

        if getattr(mf, 'with_df', None):
            self.with_df = mf.with_df
        else:
            raise KeyError('The mean-field object has no density fitting.')

    def ao2mo(self, mo_coeff=None):
        eris = _make_df_eris_incore(self, mo_coeff)
        if not self.vvvv_direct:
            nvir = self.nmo - self.nocc
            npair = nvir * (nvir + 1) // 2
            max_memory = self.max_memory - current_memory()[0]
            if 16 * npair**2 / 1e6 > .6 * max_memory:
                logger.warn(self, 'Not enough memory to store eris.vvvv')
            else:
                # a constant: its dependence on Lvv enters through the custom VJP
                Lvv = lib.unpack_tril(ops.stop_grad(eris.Lvv), filltriu=lib.SYMMETRIC)
                eris._vvvv_pair = _build_eris_vvvv(Lvv)
        return eris

class _ChemistsERIs(ccsd._ChemistsERIs):
    _dynamic_attr = {'Lvv'}
    # the dense ladder integrals are a constant cache, not compared or hashed
    # with the pytree
    _ignore_attr = ('_vvvv_pair',)

    def __init__(self, mol=None):
        super().__init__(mol=mol)
        self.naux = None
        self.Lvv = None
        self._vvvv_pair = None

    def _contract_vvvv_t2(self, mycc, t2, direct=False, out=None, verbose=None):
        assert not direct
        return _contract_vvvv_t2(mycc, self.mol, self.Lvv, t2, vvvv=self._vvvv_pair)


def _make_df_eris_incore(cc, mo_coeff=None):
    eris = _ChemistsERIs()
    eris._common_init_(cc, mo_coeff)
    nocc = eris.nocc
    nmo = eris.fock.shape[0]
    nvir = nmo - nocc
    with_df = cc.with_df
    naux = with_df.get_naoaux()

    mo = np.asarray(eris.mo_coeff)
    ijslice = (0, nmo, 0, nmo)
    eri1 = with_df._cderi
    # pylint: disable=too-many-function-args
    Lpq = _ao2mo.nr_e2(eri1, mo, ijslice, aosym='s2', mosym='s1').reshape(-1,nmo,nmo)
    Loo = Lpq[:,:nocc,:nocc].reshape(naux,-1)
    Lov = Lpq[:,:nocc,nocc:].reshape(naux,-1)
    eris.Lvv = Lvv = lib.pack_tril(Lpq[:,nocc:,nocc:])

    eris.oooo = np.dot(Loo.T, Loo).reshape(nocc,nocc,nocc,nocc)
    eris.ovoo = np.dot(Lov.T, Loo).reshape(nocc,nvir,nocc,nocc)
    ovov = np.dot(Lov.T, Lov).reshape(nocc,nvir,nocc,nvir)
    eris.ovov = ovov
    eris.ovvo = ovov.transpose(0,1,3,2)

    oovv = np.dot(Loo.T, Lvv)
    eris.oovv = lib.unpack_tril(oovv).reshape(nocc,nocc,nvir,nvir)
    eris.ovvv = np.dot(Lov.T, Lvv).reshape(nocc,nvir,-1)
    return eris
