# Change log

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
