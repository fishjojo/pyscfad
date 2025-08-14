# Copyright 2021-2025 Xing Zhang
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

from pyscfad import numpy as np

def _make_rdm1(mycc, d1, with_frozen=True, ao_repr=False, with_mf=True):
    doo, dov, dvo, dvv = d1
    nocc, nvir = dov.shape
    nmo = nocc + nvir
    dm1 = np.empty((nmo,nmo), dtype=doo.dtype)
    dm1 = dm1.at[:nocc,:nocc].set(doo + doo.conj().T)
    dm1 = dm1.at[:nocc,nocc:].set(dov + dvo.conj().T)
    dm1 = dm1.at[nocc:,:nocc].set(dm1[:nocc,nocc:].conj().T)
    dm1 = dm1.at[nocc:,nocc:].set(dvv + dvv.conj().T)
    if with_mf:
        dm1 = dm1.at[np.diag_indices(nocc)].add(2.)

    if with_frozen and mycc.frozen is not None:
        nmo = mycc.mo_occ.size
        nocc = np.count_nonzero(mycc.mo_occ > 0)
        rdm1 = np.zeros((nmo,nmo), dtype=dm1.dtype)
        if with_mf:
            rdm1 = rdm1.at[np.diag_indices(nocc)].set(2.)
        moidx = np.where(mycc.get_frozen_mask())[0]
        rdm1 = rdm1.at[moidx[:,None],moidx].set(dm1)
        dm1 = rdm1

    if ao_repr:
        mo = mycc.mo_coeff
        dm1 = np.einsum('pi,ij,qj->pq', mo, dm1, mo.conj())
    return dm1

