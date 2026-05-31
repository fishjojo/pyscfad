# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

PySCFAD is PySCF with automatic differentiation. It re-implements quantum chemistry
methods so that energies and properties are differentiable (via JAX by default) with
respect to inputs such as atomic coordinates, basis exponents, and contraction
coefficients. This enables geometry optimization, gradients, Hessians, response
properties, and gradient-based basis/parameter optimization.

The repo contains two installable packages:
- `pyscfad/` — the pure-Python differentiable chemistry library.
- `pyscfadlib/` — a C/C++/nanobind support library with optimized integral derivative
  code, built separately.

## Build & Test

PySCFAD itself is pure Python (no build step). `pyscfadlib` must be compiled or
installed (`pip install pyscfadlib`).

Build `pyscfadlib` from source:
```bash
cd pyscfadlib/pyscfadlib
mkdir build && cmake -B build && cmake --build build -j2
```
(requires LAPACK dev headers and `nanobind`; see `.github/workflows/build_pyscfadlib.sh`.)

Running tests requires both the repo root and `pyscfadlib` on `PYTHONPATH`:
```bash
export PYTHONPATH=$(pwd):$(pwd)/pyscfadlib:$PYTHONPATH
export OMP_NUM_THREADS=1
```

Run the full suite the way CI does (top-level `tests/` plus per-module tests):
```bash
pytest ./tests --verbosity=1
pytest ./pyscfad/<module>   # e.g. ./pyscfad/gto, ./pyscfad/scf, ./pyscfad/cc
```
CI iterates these modules: `scipy gto cc fci gw mp tdscf lo pbc lno` (see
`.github/workflows/run_test.sh`).

Run a single test file or test:
```bash
pytest pyscfad/gto/test/test_mole.py
pytest pyscfad/scf/test/test_hf.py::test_rhf_nuc_grad -v
```

Tests marked `_high_cost` or `_skip` are excluded by default (see `addopts` in
`pyproject.toml`). The `examples/` directory is ignored by pytest.

Lint (CI runs pylint over the whole package; config in `.pylintrc`):
```bash
pylint pyscfad
```

## Architecture

### Backend abstraction (`pyscfad/backend/`)
All array operations route through a swappable backend so the same chemistry code can
run under different AD/array engines. Selection is `jax` by default, configurable via
`PYSCFAD_BACKEND` env var or `~/.pyscfad/pyscfad.json` (allowed: `numpy`, `cupy`,
`jax`, `torch`; jax is the only fully supported one). Precision defaults to float64
(`PYSCFAD_FLOATX`).

Chemistry code should import arrays/ops through the backend, **not** directly from jax:
- `from pyscfad import numpy as np` → backend ndarray namespace (`pyscfad.backend.numpy`,
  resolved lazily via `__getattr__` to the active backend).
- `from pyscfad import ops` → AD/control-flow primitives: `stop_gradient`/`stop_grad`,
  `custom_jvp`, `jit`, `vmap`, `while_loop`, `index_update`/`index_add` (functional
  in-place updates), `to_numpy`, `is_array`.
- `pyscfad.backend.numpy.safe_sqrt` and similar guard against non-finite derivatives
  at singular points (e.g. `grad(sqrt)(0)`).

Backend implementations live in `pyscfad/backend/_jax`, `_numpy`, `_cupy`, `_torch`,
`_common`.

### Pytree nodes (differentiable objects)
Method/molecule classes are registered as JAX pytrees so JAX can trace through them.
Two mechanisms:
- `@pyscfad.util.pytree_node(leaf_names, num_args=..., exclude_aux_name=...)` — class
  decorator marking which attributes are traced leaves vs. static (auxiliary) data.
- Subclassing `pyscfad.pytree.PytreeNode` (with a `_dynamic_attr`/`_keys` list of leaf
  attributes), as `gto.Mole` and `scf.SCF` do.

Key detail: static/aux attributes are compared by JAX; changing them triggers `jax.jit`
recompilation. The traced leaves on `Mole` are `coords`, `exp`, `ctr_coeff`, `r0` —
`Mole.build()` populates these via `trace_coords`/`trace_exp`/`trace_ctr_coeff` flags
(set a flag `False` to freeze that quantity, i.e. not differentiate w.r.t. it).

PySCFAD classes typically subclass both `PytreeNode` and the corresponding PySCF class
(e.g. `class Mole(pytree.PytreeNode, pyscf_mole.Mole)`, `class SCF(pytree.PytreeNode,
pyscf_hf.SCF)`). `.to_pyscf()` strips traced attrs and returns a plain PySCF object;
inversely PySCFAD objects accept PySCF kwargs.

### Implicit differentiation (`pyscfad/implicit_diff.py`, `pyscfad/_src/implicit_diff.py`)
For iterative solvers (SCF, CCSD), differentiating through every iteration is costly and
unstable. `make_implicit_diff`, `custom_root`, and `custom_fixed_point` instead
differentiate the converged solution via the implicit function theorem. This is gated by
runtime config flags `pyscfad_scf_implicit_diff` / `pyscfad_ccsd_implicit_diff`
(default off). When enabled, the forward solve is wrapped in `stop_grad` and the
derivative is computed from the optimality/fixed-point condition. See `scf/hf.py`
(`kernel`/`SCF`) for the canonical usage.

### Runtime configuration (`pyscfad/_src/_config.py`)
Global config object `pyscfad.config` with keys prefixed `pyscfad_` (e.g.
`pyscfad_scf_implicit_diff`, `pyscfad_ccsd_implicit_diff`, `pyscfad_moleintor_opt`).
Use the `config_update` context manager to scope a change:
```python
from pyscfad import config_update
with config_update('pyscfad_scf_implicit_diff', True):
    ...
```

### Module layout
Mirrors PySCF's structure — each subpackage is a differentiable counterpart:
- `gto/` — molecule (`Mole`, `mole_lite`), molecular integrals (`moleintor*`) with
  custom JVP/VJP rules (`_moleintor_jvp.py`, `_moleintor_vjp.py`) wrapping the C library.
- `scf/` — HF/KS mean-field (`hf`, `uhf`, `rohf`, `ghf`, `hf_lite`), DIIS, CPHF,
  alternative solvers (`anderson`, `sp2`).
- `dft/`, `df/` (density fitting), `cc/` (coupled cluster), `mp/`, `fci/`, `gw/`,
  `tdscf/`, `lo/` (localization), `lno/`, `soscf/`, `geomopt/`, `prop/` (properties),
  `tools/`, `ml/`, `xtb/` (GFN1-xTB), `scipy/` (differentiable scipy helpers).
- `pbc/` — periodic boundary conditions, mirroring the molecular subpackages.
- `lib/` — Python-side numerical helpers (`numpy_helper`, `linalg_helper`, `diis`,
  `logger`).
- `experimental/` — unstable APIs.

Most `gto`/`scf`/etc. subpackages have their own `test/` directory; top-level
integration tests live in `tests/`.

### Integral derivatives via pyscfadlib
The expensive integral-derivative kernels are in `pyscfadlib` (C/C++ exposed through
nanobind). The Python integral wrappers in `pyscfad/gto/moleintor*.py` and their
JVP/VJP rules call into this library; `_pyscf_moleintor.py` bridges to PySCF's own
integral engine. `pyscfadlib` is versioned and released independently of `pyscfad`.

## Conventions

- Differentiable code must go through `pyscfad.numpy` / `pyscfad.ops`, never raw `jax`
  imports, so the backend abstraction holds.
- When adding a new differentiable class, register it as a pytree (decorator or
  `PytreeNode` subclass) and be deliberate about which attributes are traced leaves vs.
  static aux data — static data participates in jit cache keys.
- For new iterative solvers, prefer implicit differentiation over unrolling, gated by a
  `pyscfad_*_implicit_diff` config flag.
- Use `safe_*` numpy helpers (e.g. `safe_sqrt`) where derivatives can blow up at
  singularities.
