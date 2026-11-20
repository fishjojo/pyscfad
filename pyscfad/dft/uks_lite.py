# Copyright 2026 The PySCFAD Authors
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
Lightweight :mod:`~pyscfad.dft.uks` module.
"""
from pyscfad.scf import uhf_lite
from pyscfad.dft import uks
from pyscfad.dft.rks_lite import KohnShamDFT

class UKS(KohnShamDFT, uhf_lite.UHF):
    """Unrestricted Kohn-Sham DFT.
    """
    get_veff = uks.get_veff
    energy_elec = uks.energy_elec

UKSLite = UKS
