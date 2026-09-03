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

from pyscfad.mp import mp2
from pyscfad.mp import ump2
from pyscfad.mp import dfmp2
from pyscfad.mp.mp2 import RMP2
from pyscfad.mp.ump2 import UMP2

def MP2(mf, frozen=None, mo_coeff=None, mo_occ=None):
    """Create the MP2 object matching the mean-field reference ``mf``.

    Returns an instance of :class:`~pyscfad.mp.ump2.UMP2` for an
    unrestricted reference, and of :class:`~pyscfad.mp.mp2.RMP2` otherwise.
    """
    if mf.istype('UHF'):
        return UMP2(mf, frozen, mo_coeff, mo_occ)
    elif mf.istype('GHF'):
        raise NotImplementedError('GMP2 is not implemented.')
    else:
        return RMP2(mf, frozen, mo_coeff, mo_occ)
