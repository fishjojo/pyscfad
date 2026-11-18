# CLAUDE.md

PySCFAD is a differentiable quantum chemistry library built on PySCF, using JAX by
default. The repo holds three packages:

- `pyscfad` (repo root) — the pure-Python library.
- `pyscfadlib` (`pyscfadlib/`) — the C/C++ support library (custom primitives and their
  VJP rules), versioned and released independently of `pyscfad`.
- `pyscfad-cuda12-plugin` / `pyscfad-cuda13-plugin` (`pyscfadlib/plugins/cuda/`) — C++
  CUDA plugins, one per CUDA major, needed only on NVIDIA GPUs.

## Build

Install `pyscfadlib` first, then `pyscfad` from the repo root:

```bash
pip install pyscfadlib   # released wheel
pip install -e .         # editable, so tests exercise the working tree
```

- The wheel is the quick route, but the tree's `pyscfadlib` can be ahead of the release,
  and Python code silently falls back where a newer kernel is missing (e.g. `libmp_vjp`
  in `pyscfad/mp/dfmp2.py`). Build from the tree instead, with
  `pip install ./pyscfadlib`, when you change `pyscfadlib/` or work on such a path; CI
  always does.
- Source builds compile C/C++ and can run for many minutes, so let them run without a
  timeout: if your shell tool enforces one, run the build in the background and wait.
- CI installs its test dependencies, with version pins, in
  `.github/workflows/install_pyscf.sh`.
- The CUDA plugin build is described in `doc/source/getting_started/install.rst`
  ("Compiling the CUDA plugin").

## Tests

Run tests from the repo root with `python -m pytest`:

```bash
python -m pytest tests/test_scf.py   # one file
python -m pytest pyscfad/gto         # one module's in-package tests
```

- `python -m` puts the working tree first on `sys.path`. Bare `pytest` leaves `sys.path`
  alone (the repo sets `--import-mode=importlib`), so it tests whichever `pyscfad` is
  installed, which can be a stale copy or a different checkout.
- Tests whose names contain `_high_cost` or `_skip` are deselected by default (`addopts`
  in `pyproject.toml`); pass your own `-k`, e.g. `-k high_cost`, to run one. `examples/`
  is not collected.
- CI's full run is `.github/workflows/run_test.sh`: `tests/`, then `pyscfad/<module>`
  for each module in its `MODULES` list. The script puts the tree's `pyscfadlib/` on
  `PYTHONPATH`, so it needs the in-tree build from `build_pyscfadlib.sh` next to it.
- CI also runs `pylint pyscfad` (see `.pylintrc`).

### Where tests go

- If the subpackage you are testing has its own `test/` directory (e.g.
  `pyscfad/gto/test/`, `pyscfad/pbc/scf/test/`), add the test there, even when `tests/`
  also has a file for that module.
- Otherwise, including for a new module, add it under the top-level `tests/`: in
  `tests/test_{module}.py`, or in a `tests/{module}/` directory once it needs several
  files or its own fixtures (as `tests/xtb/` does).
- Within that location, extend an existing test file when one already covers the code.

### Writing tests

- Reuse existing fixtures before building a molecule by hand. Under `tests/`,
  `tests/conftest.py` provides the legacy-`Mole` factories `mol_H2`, `mol_H2O`, and
  `mol_N2`; call them, e.g. `mol_H2O(basis="sto3g")` (keywords go to
  `tests/util.make_mol`). They are not visible from an in-package `test/` directory:
  there, use its own `conftest.py` or the fixtures of the file you are extending.
- Build shared objects (molecules, parameters, reference values) in `@pytest.fixture`s,
  and share them across a directory through its `conftest.py`; `tests/xtb/conftest.py`
  shows the reference-value pattern.
- Test one behavior on the smallest system that shows it (minimal basis, two or three
  atoms), with deterministic assertions. The default suite has to stay fast.
- End the name of any slow test (large system, tight convergence, high-order
  derivatives) with `_high_cost`, e.g. `test_ccsd_gradient_high_cost`, which keeps it
  out of the default run.

## Architecture and conventions

`pyscfad` mirrors PySCF's package layout (`pyscfad.scf` ↔ `pyscf.scf`, `pyscfad.pbc` ↔
`pyscf.pbc`, ...), so look for a feature where PySCF keeps it. Additions with no PySCF
counterpart include `backend/` (array backends), `ml/` (padded `*Pad` classes for
batching), and `xtb/` (GFN1-xTB).

### Backend

- Import array operations and AD/control-flow primitives through the backend:
  `from pyscfad import numpy as np` and `from pyscfad import ops` (`stop_grad`, `jit`,
  `vmap`, `while_loop`, `custom_jvp`, `index_update`, ...). This keeps the code working
  when another backend is selected with `PYSCFAD_BACKEND` or `~/.pyscfad/pyscfad.json`
  (`jax` is the default and the only fully supported one; `numpy`, `cupy`, and `torch`
  also exist). Import from `jax` directly only for what the backend lacks, such as
  `jax.lax.custom_root`.
- Pass `dtype=np.floatx` (or `np.complexx`) when you create or cast arrays, e.g.
  `np.zeros(n, dtype=np.floatx)`. JAX always runs with x64 enabled here, so the default
  dtype stays `float64` even when `PYSCFAD_FLOATX=float32`.
- Use `np.safe_sqrt` / `np.safe_reciprocal` where a derivative is singular; for example,
  `grad(sqrt)(0)` is `inf`.

### Differentiable classes

- Write a new class that has a PySCF parent class as a plain, fully jittable class,
  modeled on `MoleLite` (`pyscfad/gto/mole_lite.py`), `SCFLite`
  (`pyscfad/scf/hf_lite.py`), `MolePad` (`pyscfad/ml/gto/mole_pad.py`), and `SCFPad`
  (`pyscfad/ml/scf/hf_pad.py`). Such a class is not a pytree: construct it inside the
  function you `jit` or `grad`, from its array arguments (see
  `examples/scf/40-rhf_jit.py`).
- Use a `@jax.tree_util.register_dataclass` dataclass for a traceable container of
  data, like `BasisArray` in `pyscfad/ml/gto/basis_array.py`.
- `pyscfad.pytree.PytreeNode` and `@pyscfad.util.pytree_node` (deprecated) are the
  legacy way to register a class as a pytree. Much existing code still uses them (e.g.
  `gto.Mole`, `scf.SCF`); leave that code as it is, and build new classes on the two
  patterns above.
- In a registered pytree, choose deliberately which attributes are traced leaves and
  which are static. Leaves are what gets differentiated; static attributes are part of
  the `jax.jit` cache key, so changing one triggers recompilation. The legacy
  `Mole.build()` selects its leaves with the `trace_coords` / `trace_exp` /
  `trace_ctr_coeff` flags, and legacy objects convert to plain PySCF ones with
  `.to_pyscf()`.

### Iterative solvers

- Differentiate an iterative solver (SCF, CCSD, ...) implicitly, through its converged
  solution, rather than by unrolling the iterations, which is costly and unstable.
- In new code, wrap the solve in `jax.lax.custom_root` with a custom `tangent_solve`,
  as `SCFLite` (`pyscfad/scf/hf_lite.py`) and `RCCSDLite` (`pyscfad/cc/rccsd.py`) do;
  this keeps the whole path jittable.
- The older helpers in `pyscfad/implicit_diff.py` (`make_implicit_diff`, `custom_root`,
  `custom_fixed_point`), switched on by the `pyscfad_scf_implicit_diff` /
  `pyscfad_ccsd_implicit_diff` config flags (default off), are for two cases only: the
  surrounding code is legacy and not fully jittable, or the linear solve is
  ill-conditioned and needs the safeguarded GMRES (`gmres_safe`, wired up through
  `gen_gmres` in `pyscfad/tools/linear_solver.py`).

### Runtime configuration and integrals

- Runtime flags live on `pyscfad.config`, with keys prefixed `pyscfad_` (defined in
  `pyscfad/_src/_config.py`). Scope a change with the context manager:
  `with config_update('pyscfad_scf_implicit_diff', True): ...`
  (`from pyscfad import config_update`).
- The integral-derivative kernels live in `pyscfadlib`. `pyscfad/gto/moleintor.py`
  dispatches to the JVP rules in `_moleintor_jvp.py` by default and to the VJP rules in
  `_moleintor_vjp.py` when `pyscfad_moleintor_opt` is on; `_pyscf_moleintor.py` bridges
  to PySCF's own integral engine.
- `pyscfad_moleintor_opt` (default off) also switches `ao2mo`, `df`, `mp`, `cc`, `lno`,
  and `scf` to matching optimized paths, so much of the code has two paths to keep
  correct: when you fix or test one, check whether the other needs the same.
