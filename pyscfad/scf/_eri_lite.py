# Copyright 2025-2026 The PySCFAD Authors
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

"""Jittable J/K contractions for packed two-electron integrals.

An ``s4``-packed ERI stores ``(ij|kl)`` for ``i >= j``, ``k >= l`` as an
``(npair, npair)`` matrix, four times smaller than the dense ``s1`` tensor,
and :func:`pyscfad.gto.moleintor_lite.getints` emits its coordinate tangents
in the same packing. Everything here (tril pack/unpack gathers, matmuls, a
scanned row contraction) is linear, so jvps pass through unchanged and
reverse mode transposes the gathers to scatter-adds -- no custom
differentiation rules are needed, and derivatives compose to any order the
integrals support.
"""
from functools import partial

from jax.lax import map as lax_map

from pyscfad import numpy as np
from pyscfad import ops
from pyscfad.gto.moleintor_lite import _tril_idx, _pair_index_matrix
from pyscfad.scf import hf

# Transient-memory budget (bytes) for the blocked K-build row scan.
VK_BLOCK_BYTES = 2**28


def dot_eri_dm(eri, dm, hermi=0, with_j=True, with_k=True):
    """Compute J/K matrices, dispatching on the ERI packing.

    ``hermi`` is accepted for API parity but not needed: both branches are
    exact for arbitrary (also non-hermitian) density matrices.
    """
    del hermi
    dm = np.asarray(dm)
    nao = dm.shape[-1]
    npair = nao * (nao + 1) // 2
    if eri.ndim == 2 and eri.shape == (npair, npair):
        return dot_eri_dm_s4(eri, dm, with_j, with_k)
    if eri.ndim == 4 or eri.size == nao**4:
        # pylint: disable-next=protected-access
        return hf._dot_eri_dm_s1(eri, dm, with_j, with_k)
    raise NotImplementedError(
        f"J/K contraction for ERI of shape {eri.shape} with nao={nao}")


@partial(ops.jit, static_argnums=(2, 3, 4))
def dot_eri_dm_s4(eri, dm, with_j=True, with_k=True, vk_block_size=None):
    """J/K matrices from an s4-packed ERI without materializing nao**4.

    J packs the symmetrized density with off-diagonal weight 2 and reduces to
    a single ``(x, npair) @ (npair, npair)`` matmul. K scans over AO row
    blocks: each step gathers the s4 rows of one block, unpacks their column
    pairs and contracts with the density, so the transient stays at
    ``O(block * nao**3)`` (block chosen to fit ``VK_BLOCK_BYTES``).
    """
    nao = dm.shape[-1]
    tril_i, tril_j = _tril_idx(nao)
    pair_idx = _pair_index_matrix(nao)

    dms = dm.reshape(-1, nao, nao)

    vj = vk = None
    if with_j:
        # (ij|kl) = (ji|kl): only the symmetric part of dm contributes.
        dm_tril = (dms + dms.transpose(0, 2, 1))[:, tril_i, tril_j]
        dm_tril = np.where(tril_i == tril_j, 0.5 * dm_tril, dm_tril)
        vj = (dm_tril @ eri)[:, pair_idx].reshape(dm.shape)

    if with_k:
        pair_idx_dev = np.asarray(pair_idx)  # the row gather index is traced
        if vk_block_size is None:
            vk_block_size = int(min(nao, max(1, VK_BLOCK_BYTES // (nao**3 * 8))))

        def _krows(i):
            rows = eri[pair_idx_dev[i, :], :]        # (j, klpair)
            w = rows[:, pair_idx]                    # (j, k, l)
            return np.einsum("jkl,xjk->xl", w, dms)  # K[i] = sum_jk (ij|kl) D[jk]

        vk = lax_map(_krows, np.arange(nao), batch_size=vk_block_size)
        vk = vk.transpose(1, 0, 2).reshape(dm.shape)
    return vj, vk
