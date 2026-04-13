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

import jax
from pyscfad import numpy as np

def make_mp2_rdm1_ie(Lia, Ljb, eia, ejb):
    naux1, nocc1, nvir1 = Lia.shape
    naux2, nocc2, nvir2 = Ljb.shape
    assert naux1 == naux2
    assert nvir1 == nvir2
    naux = naux1
    nvir = nvir1

    @jax.checkpoint
    def fn(carry, x):
        dmvv, dmoo = carry
        La, ea = x
        buf = np.dot(La.T, Ljb.reshape(naux,-1)).reshape(nvir, nocc2, nvir)
        t2i = buf / (ea[:,None,None] + ejb[None,:,:])
        dmvv +=  np.dot(t2i.reshape(nvir, -1), t2i.reshape(nvir, -1).T)
        dmvv -= .5 * np.einsum('ajc,cjb->ab', t2i, t2i)
        dmvv += np.dot(t2i.reshape(-1,nvir).T, t2i.reshape(-1,nvir))
        dmvv -= .5 * np.einsum('cja,bjc->ab', t2i, t2i)

        dmoo += np.einsum('aib,ajb->ij', t2i, t2i)
        dmoo -= .5 * np.einsum('aib,bja->ij', t2i, t2i)
        dmoo += np.einsum('bia,bja->ij', t2i, t2i)
        dmoo -= .5 * np.einsum('bia,ajb->ij', t2i, t2i)
        return (dmvv, dmoo), None

    (dmvv, dmoo), _ = jax.lax.scan(fn, (np.zeros((nvir, nvir)), np.zeros((nocc2,nocc2))),
                                   (Lia.transpose(1,0,2), eia))
    return dmvv, dmoo
