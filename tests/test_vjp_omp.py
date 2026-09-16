# Copyright 2021-2025 The PySCFAD Authors
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

'''Thread safety and 64-bit indexing of the pyscfadlib vjp drivers.

The failures covered here depend on the OpenMP thread count and on array
sizes that cannot be chosen once the process is running, and a regression
aborts the interpreter rather than raising.  Each test therefore re-executes
a small script in a subprocess with its own ``OMP_NUM_THREADS``.
'''

import os
import subprocess
import sys
import pytest

# skip code the helper scripts use to report that the machine cannot host
# the (large, mostly untouched) allocation a test needs
NO_MEMORY = 77

ATOM_H2O = 'O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587'


def _run(script, nthreads):
    env = dict(os.environ)
    env['OMP_NUM_THREADS'] = str(nthreads)
    env['PYTHONPATH'] = os.pathsep.join(p for p in sys.path if p)
    proc = subprocess.run([sys.executable, '-c', script], env=env,
                          capture_output=True, text=True, timeout=900,
                          check=False)
    if proc.returncode == NO_MEMORY:
        pytest.skip('not enough address space for this test')
    assert proc.returncode == 0, (
        f'subprocess failed (returncode={proc.returncode})\n'
        f'--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}')
    return proc.stdout


@pytest.fixture
def int3c_ij_r0_vjp_script():
    # GTOnr3c_ij_r0_vjp lets thread 0 accumulate straight into the shared
    # output while the other threads reduce their private buffers into it;
    # without a barrier in between whole partial sums are dropped.
    return f'''
import ctypes
import numpy
from pyscf import gto, df, lib
from pyscf.gto.moleintor import ascint3, make_loc
from pyscfadlib import libcgto_vjp as libcgto

mol = gto.M(atom="{ATOM_H2O}", basis="ccpvdz", verbose=0)
auxmol = df.addons.make_auxmol(mol)
nao, naux = mol.nao, auxmol.nao
natm = mol.natm
comp = 3

pmol = mol + auxmol
atm = numpy.asarray(pmol._atm, dtype=numpy.int32, order="C")
bas = numpy.asarray(pmol._bas, dtype=numpy.int32, order="C")
env = numpy.asarray(pmol._env, dtype=numpy.double, order="C")
intor = ascint3(mol._add_suffix("int3c2e_ip1"))
ao_loc = numpy.asarray(make_loc(bas, intor), dtype=numpy.int32, order="C")
shls_slice = (0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas+auxmol.nbas)

rng = numpy.random.default_rng(12345)
ybar = numpy.asarray(rng.standard_normal((nao*(nao+1)//2, naux)),
                     order="F", dtype=numpy.double)

def one_call():
    vjp = numpy.zeros((natm, comp), order="C", dtype=numpy.double)
    libcgto.GTOnr3c_ij_r0_vjp(
        getattr(libcgto, intor), libcgto.GTOnr3c_ij_r0_vjp_s2ij,
        vjp.ctypes.data_as(ctypes.c_void_p),
        ybar.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(comp), (ctypes.c_int*6)(*shls_slice),
        ao_loc.ctypes.data_as(ctypes.c_void_p), lib.c_null_ptr(),
        atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(atm)),
        ctypes.c_int(natm),
        bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(bas)),
        env.ctypes.data_as(ctypes.c_void_p))
    return vjp

ref = one_call()
spread = max(abs(one_call() - ref).max() for _ in range(7))
print(f"scale {{abs(ref).max():.6e}}")
print(f"spread {{spread:.6e}}")
'''


@pytest.fixture
def int2e_r0_vjp_script():
    # same missing barrier in GTOnr2e_fill_r0_vjp
    return f'''
import numpy
from pyscf import gto
from pyscfad.gto import _moleintor_vjp

mol = gto.M(atom="{ATOM_H2O}", basis="ccpvdz", verbose=0)
npair = mol.nao*(mol.nao+1)//2
rng = numpy.random.default_rng(12345)
ybar = numpy.asarray(rng.standard_normal((npair, npair)), order="C")

def one_call():
    return _moleintor_vjp.getints4c_coords_bwd(
        mol._add_suffix("int2e"), None, 1, "s4", None, mol, ybar.copy())

ref = one_call()
spread = max(abs(one_call() - ref).max() for _ in range(3))
print(f"scale {{abs(ref).max():.6e}}")
print(f"spread {{spread:.6e}}")
'''


@pytest.fixture
def df_vk_vjp_script():
    # the per-thread buffer table used to be a fixed-size stack array
    return '''
import ctypes
import numpy
from pyscfadlib import libcvhf_vjp as libvhf

nao, naux = 24, 64
npair = nao*(nao+1)//2
rng = numpy.random.default_rng(0)
eri_tril = numpy.ascontiguousarray(rng.standard_normal((naux, npair)))
eri_tril_bar = numpy.zeros((naux, npair))
buf1 = numpy.ascontiguousarray(rng.standard_normal((naux, nao, nao)))
dm = numpy.asarray(rng.standard_normal((nao, nao)), order="F")
vk_bar = numpy.asarray(rng.standard_normal((nao, nao)), order="F")
dm_bar = numpy.zeros((nao, nao), order="F")

libvhf.df_vk_vjp(
    eri_tril_bar.ctypes.data_as(ctypes.c_void_p),
    dm_bar.ctypes.data_as(ctypes.c_void_p),
    vk_bar.ctypes.data_as(ctypes.c_void_p),
    buf1.ctypes.data_as(ctypes.c_void_p),
    eri_tril.ctypes.data_as(ctypes.c_void_p),
    dm.ctypes.data_as(ctypes.c_void_p),
    ctypes.c_int(naux), ctypes.c_int(nao))
print(f"dm_bar {numpy.linalg.norm(dm_bar):.10f}")
print(f"eri_bar {numpy.linalg.norm(eri_tril_bar):.10f}")
'''


@pytest.fixture
def int3c_fill_s2ij_script():
    # GTOnr3c_fill_s2ij strides the output by nij*naok between components;
    # that product exceeds 2**31 for large DF systems and used to be
    # truncated to int, sending the writes far outside the array.
    return f'''
import ctypes
import sys
import numpy
from pyscf import gto, lib
from pyscf.gto.moleintor import ascint3, getints_by_shell, make_loc
from pyscfadlib import libcgto_vjp as libcgto

# an H chain in a single-function basis, split into a bra block of NI
# shells and an auxiliary block of NK shells, so that nij*naok > 2**31
NI, NK = 1000, 4291
nij = NI*(NI+1)//2
assert nij*NK > 2**31
comp = 3

mol = gto.M(atom=[["H", (0., 0., 1.5*i)] for i in range(NI+NK)],
            basis="sto-3g", spin=None, verbose=0)
assert mol.nbas == NI+NK

intor = ascint3(mol._add_suffix("int3c2e_ip2"))
atm = numpy.asarray(mol._atm, dtype=numpy.int32, order="C")
bas = numpy.asarray(mol._bas, dtype=numpy.int32, order="C")
env = numpy.asarray(mol._env, dtype=numpy.double, order="C")
ao_loc = numpy.asarray(make_loc(bas, intor), dtype=numpy.int32, order="C")
shls_slice = (0, NI, 0, NI, NI, NI+NK)

# (comp, naok, nij) in C order, left uninitialized so that the pages the
# driver never touches are never faulted in
try:
    out = numpy.ndarray(comp*NK*nij, dtype=numpy.double)
except (MemoryError, ValueError):
    sys.exit({NO_MEMORY})
buf = numpy.empty(1 << 20, dtype=numpy.double)

# jobid 0 only: shells 0:8 of the bra block against the first aux shell
libcgto.GTOnr3c_fill_s2ij(
    getattr(libcgto, intor),
    out.ctypes.data_as(ctypes.c_void_p),
    buf.ctypes.data_as(ctypes.c_void_p),
    ctypes.c_int(comp), ctypes.c_int(0),
    (ctypes.c_int*6)(*shls_slice),
    ao_loc.ctypes.data_as(ctypes.c_void_p), lib.c_null_ptr(),
    atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(atm)),
    bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(bas)),
    env.ctypes.data_as(ctypes.c_void_p))

# each component must land exactly nij*naok apart, i.e. the first bra
# pair of the first aux shell reproduces that single shell triple
got = numpy.array([out[ic*NK*nij] for ic in range(comp)])
ref = getints_by_shell(intor, (0, 0, NI), atm, bas, env, comp).ravel()
print(f"nijk {{nij*NK}}")
print(f"err {{abs(got-ref).max()/max(abs(ref).max(), 1e-300):.3e}}")
'''


def _spread(stdout):
    out = dict(line.split() for line in stdout.strip().splitlines())
    return float(out['spread']), float(out['scale'])


def test_int3c_ij_r0_vjp_thread_safe(int3c_ij_r0_vjp_script):
    spread, scale = _spread(_run(int3c_ij_r0_vjp_script, 16))
    # only the order of the final reduction may change
    assert spread < 1e-9 * max(scale, 1.)


def test_int2e_r0_vjp_thread_safe(int2e_r0_vjp_script):
    spread, scale = _spread(_run(int2e_r0_vjp_script, 16))
    assert spread < 1e-9 * max(scale, 1.)


def test_df_vk_vjp_many_threads(df_vk_vjp_script):
    ref = _run(df_vk_vjp_script, 1)
    # a fixed 128-entry per-thread buffer table overflowed past 128 threads
    assert _run(df_vk_vjp_script, 200) == ref


def test_int3c_fill_s2ij_stride_above_int32(int3c_fill_s2ij_script):
    out = dict(line.split() for line in
               _run(int3c_fill_s2ij_script, 1).strip().splitlines())
    assert int(out['nijk']) > 2**31
    assert float(out['err']) < 1e-12
