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

import os
import sys
import json
import textwrap
import subprocess
import numpy
import pytest
from pyscfad import numpy as np

# Sentinel prefixing the single JSON result line emitted by an FP32 subprocess,
# so it can be picked out of stdout regardless of any framework/JAX noise.
_FP32_SENTINEL = "@@FP32_RESULT@@"

_FP32_PREAMBLE = """\
import json
import numpy
import jax
from pyscfad import numpy as np

IN = json.loads(r'''{inputs_json}''')

def emit(**kw):
    print({sentinel!r} + json.dumps(kw))

"""

@pytest.fixture
def run_fp32():
    """Run a snippet under ``PYSCFAD_FLOATX=float32`` in a fresh subprocess.

    ``np.floatx``, the eigh ``DEG_THRESH``, padded-basis sentinel exponents and
    every other module-level constant are frozen *at import time* from the
    global ``PYSCFAD_FLOATX`` setting. An in-process swap of ``np.floatx`` can
    therefore only fix some of them, leaving the rest at their float64 values --
    which is why the FP32 path must be exercised in a subprocess that imports
    the whole stack with float32 as the default working precision, so all
    constants take their correct float32 values.

    ``body`` is Python source run after a small preamble that imports
    ``numpy``/``jax``/``pyscfad as np`` and exposes the keyword ``inputs`` as the
    dict ``IN`` (JSON-round-tripped; numpy/JAX arrays are accepted and arrive as
    nested lists) plus an ``emit(**kw)`` helper. The body must call ``emit``
    exactly once with JSON-serialisable results; ``run_fp32`` returns that dict.
    """
    def run(body, **inputs):
        inputs_json = json.dumps(
            inputs, default=lambda o: numpy.asarray(o).tolist())
        script = _FP32_PREAMBLE.format(
            inputs_json=inputs_json, sentinel=_FP32_SENTINEL)
        script += textwrap.dedent(body)

        env = dict(os.environ)
        env["PYSCFAD_FLOATX"] = "float32"
        proc = subprocess.run(
            [sys.executable, "-c", script],
            env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            raise AssertionError(
                f"FP32 subprocess failed (returncode {proc.returncode}).\n"
                f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}")
        for line in reversed(proc.stdout.splitlines()):
            if line.startswith(_FP32_SENTINEL):
                return json.loads(line[len(_FP32_SENTINEL):])
        raise AssertionError(
            "FP32 subprocess produced no result line.\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}")
    return run

@pytest.fixture
def H2O_GFN1_ref():
    numbers = np.array([8, 1, 1])
    coords = np.array(
        [
            [0.00000,  0.00000,  0.00000],
            [1.43355,  0.00000, -0.95296],
            [1.43355,  0.00000,  0.95296],
        ]
    )
    e = -5.7231173051309865
    g = np.array(
        [
            [ 3.93688356e-02, 0.,  0.],
            [-1.96844178e-02, 0.,  1.03173514e-01],
            [-1.96844178e-02, 0., -1.03173514e-01],
        ]
    )
    mu = np.array([3.77507247, 0., 0.])
    alpha = np.diag(np.array([7.33970468, 1.07102937e-01, 4.97202824]))
    yield (numbers, coords, e, g, mu, alpha)

@pytest.fixture
def NH3_GFN1_ref():
    numbers = np.array([7, 1, 1, 1])
    coords = np.array(
        [
            [-0.80650, -1.00659,  0.02850],
            [-0.50540, -0.31299,  0.68220],
            [ 0.00620, -1.41579, -0.38500],
            [-1.32340, -0.54779, -0.69350],
        ]
    ) / 0.52917721067121
    e = -4.82989868
    g = np.array(
        [
            [ 0.00307771,  0.00378012, -0.00250677],
            [-0.00197155, -0.00543102, -0.00679491],
            [-0.00681684,  0.00491176,  0.0032046 ],
            [ 0.00571069, -0.00326086,  0.00609709],
        ]
    )
    mu = np.array([1.14764393,  1.42879716, -0.92636369])
    alpha = np.array(
        [
            [ 8.40057146, -3.05976767,  1.98425083],
            [-3.05976767,  7.04997063,  2.47067676],
            [ 1.98425083,  2.47067676,  9.25966469],
        ]
    )
    yield (numbers, coords, e, g, mu, alpha)
