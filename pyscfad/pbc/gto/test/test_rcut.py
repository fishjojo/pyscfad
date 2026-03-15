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

import dataclasses
import pytest
import numpy
from pyscf.pbc import gto as pyscf_gto
from pyscfad.pbc import gto
from pyscfad.pbc.gto.cell import estimate_rcut

@dataclasses.dataclass
class Diamond:
    atom = """
        C 0.875 0.875 0.875
        C 0.25 0.25 0.25
    """
    a = """
        4.136576868, 0.000000000, 2.388253772
        1.378858962, 3.900002074, 2.388253772
        0.000000000, 0.000000000, 4.776507525
    """
    unit = "B"
    fractional = True

@pytest.fixture
def pyscf_cell():
    diamond = Diamond()
    cell = pyscf_gto.M(
        atom=diamond.atom,
        a=diamond.a,
        unit=diamond.unit,
        fractional=diamond.fractional,
        precision=1e-20,
        basis = "gth-tzv2p",
        pseudo="gth-lda",
        mesh = [15]*3,
        verbose = 0,
    )
    return cell

@pytest.fixture
def ad_cell():
    diamond = Diamond()
    cell = gto.Cell()
    cell.atom = diamond.atom
    cell.a = diamond.a
    cell.unit = diamond.unit
    cell.fractional = diamond.fractional
    #cell.precision = 1e-20
    cell.basis = "gth-tzv2p"
    cell.pseudo = "gth-lda"
    cell.mesh = [15]*3
    cell.verbose = 0
    cell.build()
    return cell

def test_rcut(pyscf_cell, ad_cell):
    cell = pyscf_cell
    kpts = cell.make_kpts([2,2,2])
    s0 = numpy.asarray(cell.pbc_intor("int1e_ovlp", hermi=1, kpts=kpts))
    t0 = numpy.asarray(cell.pbc_intor("int1e_kin", hermi=1, kpts=kpts))

    cell = ad_cell
    for i in range(10):
        prec = 1e-12 * 10**i
        rcut1 = max([cell.bas_rcut(ib, prec) for ib in range(cell.nbas)])
        rcut2 = estimate_rcut(cell, prec)
        assert abs(rcut1 - rcut2) < 1e-3
        cell.rcut = rcut2
        s1 = numpy.asarray(cell.pbc_intor("int1e_ovlp", hermi=1, kpts=kpts))
        t1 = numpy.asarray(cell.pbc_intor("int1e_kin", hermi=1, kpts=kpts))
        #print(prec, cell.rcut, "error = ", numpy.linalg.norm(s1-s0), numpy.linalg.norm(t1-t0))
        assert abs(s1-s0).max() < prec
        assert abs(t1-t0).max() < prec
