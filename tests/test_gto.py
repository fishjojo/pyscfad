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

"""Tests for pyscfad.gto
"""
import pickle
import warnings
import numpy
import jax
from pyscfad import gto
from pyscfad import ops

TRACED = ('coords', 'exp', 'ctr_coeff')

def _traced_mol(mol_H2):
    return mol_H2(trace_exp=True, trace_ctr_coeff=True)

def test_mole_dumps(mol_H2):
    mol = _traced_mol(mol_H2)
    with warnings.catch_warnings():
        # the traced attributes must not be dropped as non-serializable
        warnings.simplefilter('error', UserWarning)
        molstr = mol.dumps()

    mol1 = gto.Mole.loads(molstr)
    assert isinstance(mol1, gto.Mole)
    for key in TRACED:
        val = getattr(mol1, key)
        assert ops.is_array(val)
        assert abs(numpy.asarray(val) - numpy.asarray(getattr(mol, key))).max() < 1e-12

def test_mole_pickle(mol_H2):
    mol = _traced_mol(mol_H2)
    mol1 = pickle.loads(pickle.dumps(mol))
    for key in TRACED:
        assert abs(numpy.asarray(getattr(mol1, key))
                   - numpy.asarray(getattr(mol, key))).max() < 1e-12

def test_mole_dumps_traced(mol_H2):
    # tracers hold no concrete value; dumps drops them instead of failing
    mol = _traced_mol(mol_H2)
    @jax.jit
    def fn(mol):
        mol.dumps()
        return mol.coords.sum()
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        assert abs(fn(mol) - numpy.asarray(mol.coords).sum()) < 1e-12
