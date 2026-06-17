# CLAUDE.md

## What This Is

This is the PySCFAD repository.
The repo contains three packages:
- **pyscfad** - the pure-Python differentiable (via JAX by default) quantum chemistry library (see `pyproject.toml`).
- **pyscfadlib** - a C/C++ support library with custom primitives and vjp rules (see `pyscfadlib/pyproject.toml`).
- **pyscfad-cuda12-plugin** / **pyscfad-cuda13-plugin** - C++ plugins (one per CUDA major) interfacing with third-party CUDA libraries, built with CMake (see `pyscfadlib/plugins/cuda/`).

## Building

### Build Commands

- **pyscfadlib**: `pip install pyscfadlib`
  - **Build from source**:
    ```bash
    cd pyscfadlib
    pip install .
    ```

- **pyscfad-cuda12-plugin** / **pyscfad-cuda13-plugin**: `pip install pyscfad-cuda12-plugin`
  (or `pyscfad-cuda13-plugin`)
  - **CRITICAL**: only needed when running on NVIDIA GPUs.
  - **Build from source** (CMake): needs a matching CUDA toolkit on `PATH`
    (CUDA 12.8+ for the cuda12 wheel, 13.x for cuda13) plus the `cmake`, `nanobind`,
    `jax`, and `build` Python packages. From `pyscfadlib/`:
    ```bash
    python plugins/cuda/build_plugin.py --cuda-major 13   # or --cuda-major 12
    pip install dist/pyscfad_cuda13_plugin*.whl
    ```
    `build_plugin.py` drives `plugins/cuda/CMakeLists.txt`, which builds the `_solver`
    (cuSOLVER) and `_cuint` nanobind modules and fetches the `cuint` kernels from GitHub.
    Device archs default to up to `sm_120` for the CUDA major; override with `--cuda-arch`.

- **pyscfad**: `pip install pyscfad`
  - **Build from source** (requires pyscfadlib build first; run from the repo root):
    ```bash
    pip install .
    ```

**CRITICAL**: do not set a timeout when building from source.

## Testing

### Running Tests

- **Single test file**: `pytest tests/test_scf.py`
- **Single module tests**: `pytest pyscfad/gto`
- **Full suite tests**: see `.github/workflows/run_test.sh`

Tests marked `_high_cost` or `_skip` are excluded by default (see `addopts` in
`pyproject.toml`). The `examples/` directory is ignored by pytest.

### Test Organization

**Default: add your test to existing test files unless new modules are added.**

When adding tests for a new module, place them under `tests/` by size:
- **Small** (a handful of tests): add a single file `tests/test_{module_name}.py`.
  - Example: tests for the `scf` module live in `tests/test_scf.py`; tests for `dft`
    live in `tests/test_dft.py`.
- **Larger** (multiple files / fixtures): create a subdirectory
  `tests/{module_name}/test_*.py`.
  - Example: the `xtb` module has `tests/xtb/` with `test_xtb.py`, `test_kxtb.py`,
    and `test_xtb_pad.py`.

Note the legacy layout: many modules also keep tests in-package at
`pyscfad/{module_name}/test/test_*.py` (e.g. `pyscfad/gto/test/`, `pyscfad/cc/test/`,
`pyscfad/df/test/`). These still run, but prefer the top-level `tests/` locations above
for new tests.

### Writing Tests

- **Reuse shared fixtures from `tests/conftest.py` first.** It already provides common
  molecule fixtures (e.g. `mol_H2`, `mol_H2O`, `mol_N2`) built via `tests/util.make_mol`.
  Prefer these over constructing molecules by hand so systems stay consistent and small.
- **Use `@pytest.fixture` whenever possible.** For anything not covered by
  `tests/conftest.py`, build shared objects (molecules, parameters, reference values) in
  fixtures rather than reconstructing them inside each test, and share them across a
  module's directory via a local `conftest.py` (see `tests/xtb/conftest.py` for the
  reference-value pattern, and the `setup` fixture in `tests/xtb/test_xtb.py`).
- **Keep tests small and fast.** A test should exercise one behavior with the smallest
  system that demonstrates it (e.g. a minimal basis, a couple of atoms). The default
  suite must stay quick — prefer cheap molecules and tight, deterministic assertions.
- **Name expensive tests with the `_high_cost` suffix.** Any test that is slow or
  resource-heavy (large systems, deep convergence, big derivatives) must end its name
  with `_high_cost`, e.g. `def test_ccsd_gradient_high_cost():`. These are excluded from
  the default run via `addopts` in `pyproject.toml` (`_skip` is similarly excluded). Keep
  the bulk of coverage in fast, default-run tests.

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

**New code: prefer fully jittable, plain implementations.** Write a plain class or a
`@jax.tree_util.register_dataclass` dataclass that is fully jittable, following the
canonical `*Lite` / `*Pad` implementations:
- `MoleLite` (`pyscfad/gto/mole_lite.py`), `MolePad` (`pyscfad/ml/gto/mole_pad.py`)
- `SCFLite` (`pyscfad/scf/hf_lite.py`), `SCFPad` (`pyscfad/ml/scf/hf_pad.py`)

These are the templates to copy when adding a new differentiable class.

**Legacy mechanisms (do not use for new code):**
- `@pyscfad.util.pytree_node(...)` — **deprecated; do not use.**
- `pyscfad.pytree.PytreeNode` — legacy base class, retained only for CPU,
  non-fully-jittable code paths (e.g. the original `gto.Mole`, `scf.SCF`, which subclass
  both `PytreeNode` and the corresponding PySCF class). Do not build new code on it.

Key detail (applies to any mechanism): static/aux attributes are compared by JAX, so
changing them triggers `jax.jit` recompilation, while traced leaves are the
differentiated quantities. On the legacy `Mole` the traced leaves are `coords`, `exp`,
`ctr_coeff`, `r0` — `Mole.build()` populates these via
`trace_coords`/`trace_exp`/`trace_ctr_coeff` flags (set a flag `False` to freeze that
quantity, i.e. not differentiate w.r.t. it). Be deliberate about which attributes are
traced leaves vs. static aux data. The legacy classes also provide `.to_pyscf()` (strips
traced attrs, returns a plain PySCF object) and accept PySCF kwargs.

### Implicit differentiation

For iterative solvers (SCF, CCSD), differentiating through every iteration is costly and
unstable. Instead, differentiate the converged solution via the implicit function
theorem: the forward solve runs under `stop_grad` and the derivative is recovered from
the optimality/fixed-point condition.

**New code: prefer `jax.lax.custom_root`.** Wrap the converged solve directly with
`custom_root` (plus a custom `tangent_solve`), as `SCFLite` does in `pyscfad/scf/hf_lite.py`
(see the `custom_root(root_fn, dm, oracle, tangent_solve, ...)` call). This keeps the
whole path fully jittable.

**Legacy / special-case path (`pyscfad/implicit_diff.py`, `pyscfad/_src/implicit_diff.py`):**
`make_implicit_diff`, `custom_root`, and `custom_fixed_point` here are the older helpers,
gated by config flags `pyscfad_scf_implicit_diff` / `pyscfad_ccsd_implicit_diff`
(default off); see `scf/hf.py` (`kernel`/`SCF`) for that usage. Use this path only when:
- the surrounding code is legacy / not fully jittable, or
- the implicit-differentiation linear solve is ill-conditioned/ill-defined and needs the
  safeguarded GMRES (`gmres_safe` in `pyscfad/scipy/sparse/linalg.py`, wired up via
  `gen_gmres` in `pyscfad/tools/linear_solver.py`) rather than a plain `tangent_solve`.

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
  `ao2mo/` (integral transformation), `tools/`, `ml/`, `xtb/` (GFN1-xTB),
  `scipy/` (scipy helpers).
- `pbc/` — periodic boundary conditions, mirroring the molecular subpackages.
- `lib/` — Python-side numerical helpers (`numpy_helper`, `linalg_helper`, `diis`,
  `logger`).
- `experimental/` — unstable APIs.

Many subpackages still carry a legacy in-package `test/` directory (e.g.
`pyscfad/gto/test/`); new tests should instead go under the top-level `tests/` tree (see
**Test Organization** above).

### Integral derivatives via pyscfadlib

The expensive integral-derivative kernels are in `pyscfadlib` (C/C++ exposed through
nanobind). The Python integral wrappers in `pyscfad/gto/moleintor*.py` and their
JVP/VJP rules call into this library; `_pyscf_moleintor.py` bridges to PySCF's own
integral engine. `pyscfadlib` is versioned and released independently of `pyscfad`.

## Conventions

- Differentiable code should go through `pyscfad.numpy` / `pyscfad.ops` whenever
  possible, so the backend abstraction holds.
- When adding a new differentiable class, make it a fully jittable plain class or
  `@jax.tree_util.register_dataclass` dataclass following `MoleLite`/`SCFLite` (do **not**
  use the deprecated `pyscfad.util.pytree_node` or the legacy `PytreeNode` base). Be
  deliberate about which attributes are traced leaves vs. static aux data — static data
  participates in jit cache keys.
- For new iterative solvers, prefer implicit differentiation over unrolling, gated by a
  `pyscfad_*_implicit_diff` config flag.
- Use `safe_*` numpy helpers (e.g. `safe_sqrt`) where derivatives can blow up at
  singularities.
