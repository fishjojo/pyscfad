# Change log

## pyscfad 0.2.0 (December 18, 2025)

* Changes
  * Add support to Python 3.14.
  * Update for compatibility with jax 0.8.

* Bug fixes
  * Fix stress tensor

## pyscfad 0.1.11 (August 13, 2025)

* Changes
  * Drop support to Python 3.10.
  * Update for compatibility with pyscf 2.10.
  * Update for compatibility with jax 0.7.
  * Add int3c2e auxiliary basis response.
  * Update rcut estimation for lattice sum.

* Bug fixes
  * Fix optimized DF-CCSD(T)
  * Fix implicit differentiation for PBC methods

## pyscfad 0.1.10 (April 12, 2025)

* Changes
  * Fix 0.1.9 wheel

## pyscfad 0.1.9 (April 11, 2025)

* Changes
  * Drop support to Python 3.9.
  * Refactor code for dynamic polarizability.
  * Update for compatibility with pyscf 2.7, and 2.8.
  * Add unrestricted KS-DFT.
  * Add CUDA support.

* Bug fixes
  * Fix 1e integral high-order derivatives

## pyscfad 0.1.8 (October 30, 2024)

* Changes
  * Add support to Python 3.13.
  * Add `scipy.logm` ensuring real results when possible.

* Bug fixes
  * Fix `logger.flush` not catching certain formats.

## pyscfad 0.1.7 (July 26, 2024)

* Changes
  * Drop support to Python 3.8.
  * Add `pytree.PytreeNode`.
  * Adapt class definition using `PytreeNode`.
  * Add `util.to_pyscf`.
  * Update `gto.mole._moleintor_jvp` for shell slicing.
  * Add `scipy.linalg.eigh` JAX primitive for CPU.
  * Improve `lo.orth.lowdin` gradient stability.

* Bug fixes
  * Fix example `dft/01-DFT+U.py`.
  * Fix `scf.addons.canonical_orth_`.
  * Fix GMRES solver for scipy versions before 1.12.

## pyscfad 0.1.6 (June 27, 2024)
* Changes
  * Update GMRES solver according to scipy updates.

* Bug fixes
  * Fix minor import error in `lo.pipek`.
  * Fix PBC lattice response.
  * Fix pyscfadlib runtime link issue.

## pyscfad 0.1.5 (June 22, 2024)
* Changes
  * pyscfad is now compatable with pyscf 2.6.
  * Add `backend` module (experimental).
  * Add GCCSD(T).
  * Add interface to Jabobi sweep for Pipek-Mezey localization.

* Bug fixes
  * Fix LRC hybrid density functionals.

## pyscfad 0.1.4 (Mar 5, 2024)
* Changes
  * pyscfad is now compatable with pyscf 2.3.
  * Drop support for python 3.7.
  * Drop dependence on jaxopt.
  * Update JAX custom pytree node auxiliary data for safe comparison.
  * Add `pyscfadlib`, fast C codes for custom VJPs.
  * Add dynamic configuration.
  * Allow `implicit_diff` to use preconditioners.
  * Improve `scipy.linalg.eigh`.
  * Add `scipy.linalg.svd`.
  * Improve `lib.unpack_tril`, `lib.pack_tril`.
  * Refactor `gto.moleintor`.
  * Add fast VJP for molecular integrals.
  * Improve `gto.eval_gto` performance.
  * Add `gto.eval_gto` gradient w.r.t. `Mole.ctr_coeff` and `Mole.exp`.
  * Avoid `df.DF` to create temperary files.
  * Optimize `df.df_jk`.
  * Add `scf.cphf`.
  * Add `ao2mo._ao2mo`.
  * Add `lo.iao`, `lo.boys`, `lo.pipek`, `lo.orth`.
  * Add `geomopt`.
  * Add `mp.dfmp2`, MP2 one-RDM.
  * Consider permutation symmetry for CCSD.
  * Disable `jit` for CCSD which causes memory leak.
  * Simplify implementation of `cc.ccsd_t_slow`.
  * Add optimized `cc.ccsd_t`.
  * Add iterative CCSD(T) solver.
  * Add `cc.dfccsd`, `cc.dcsd`.
  * Add `tools.timer`.

* Bug fixes
  * Fix integral derivatives w.r.t. `Mole.ctr_coeff` and `Mole.exp`.

## pyscfad 0.1.3 (Sep 13, 2023)
* Bug fixes
  * Fix installation issues.

## pyscfad 0.1.2 (Mar 13, 2023)
* Changes
  * Add AD support for ROHF.

## pyscfad 0.1.1 (Jan 25, 2023)
* Changes
  * pyscfad is now compatable with pyscf 2.1.

## pyscfad 0.1.0 (Aug 3, 2022)
* Changes
  * First release.
