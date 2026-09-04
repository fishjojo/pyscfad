# Method coverage

pyscfad re-implements a subset of pyscf's methods so that they can be
differentiated. This page lists what that subset currently is, so that a
missing method can be told apart from a bug.

The tables below are a snapshot of pyscfad 0.3.3. Anything not listed
either has no differentiable counterpart in pyscfad, or is only usable
through `mol.to_pyscf()` / `mf.to_pyscf()`, which strips the traced
attributes and returns a plain pyscf object.

Most of the methods below are built on the legacy `gto.Mole`. That path
is gradually being transitioned to the fully jittable `gto.MoleLite` path
(the `*Lite` classes listed in the tables), and new methods may not be
added to the legacy path. New code is encouraged to use `gto.MoleLite`
and its `*Lite` methods wherever a counterpart exists, falling back to
the legacy `gto.Mole` only for the methods that have not been ported yet.

## Mean field

| Method | Class | Notes |
| --- | --- | --- |
| RHF | `scf.RHF` | |
| UHF | `scf.UHF` | |
| ROHF | `scf.ROHF` | |
| GHF | `scf.GHF` | |
| RKS | `dft.RKS` | |
| UKS | `dft.UKS` | |
| Density fitting | `mf.density_fit()`, `df.DF` | RHF, UHF, ROHF, GHF, RKS, UKS |
| Fully jittable RHF | `scf.hf_lite.SCFLite` | used with `gto.MoleLite` |

ROKS is not implemented.

## Correlated methods

| Method | Class | Notes |
| --- | --- | --- |
| RMP2 | `mp.RMP2` | `mp.MP2(mf)` dispatches on the reference |
| UMP2 | `mp.UMP2` | |
| DF-RMP2 | `mp.dfmp2.MP2` | |
| RCCSD | `cc.RCCSD` | |
| DF-RCCSD | `cc.dfccsd.RCCSD` | |
| RCCSD(T) | `mycc.ccsd_t()` | `cc.ccsd_t` (C kernels) or `cc.ccsd_t_slow` |
| RDCSD | `cc.dfdcsd.RDCSD` | |
| Fully jittable RCCSD | `cc.RCCSDLite` | used with `scf.hf_lite.SCFLite` |
| LNO-MP2 / LNO-CCSD / LNO-CCSD(T) | `lno.LNOMP2`, `lno.LNOCCSD`, `lno.LNOCCSD_T` | |
| direct RPA | `gw.rpa.RPA` | |
| CIS | `tdscf.CIS` | RHF reference |
| FCI | `fci.fci_slow` | functions only, no class |

GMP2, and the unrestricted and generalized coupled-cluster methods
(UCCSD, UCCSD(T), GCCSD, GCCSD(T)) are not implemented. Neither are
CISD, CASSCF/CASCI, and the `GW` methods other than direct RPA.

## Other modules

| Module | Contents |
| --- | --- |
| `pyscfad.lo` | Boys (`lo.boys.Boys`) and Pipek-Mezey (`lo.pipek.PM`) localization |
| `pyscfad.prop` | RHF polarizability (`prop.polarizability.rhf`), thermochemistry |
| `pyscfad.geomopt` | geometry optimization through `geometric` |
| `pyscfad.xtb` | GFN1-xTB, molecular and periodic, with QM/MM |
| `pyscfad.ml` | padded `ml.gto.MolePad` / `ml.scf.SCFPad` for batched machine learning |

## Periodic boundary conditions

`pyscfad.pbc` mirrors a smaller part of `pyscf.pbc`:

| Method | Class | Notes |
| --- | --- | --- |
| RHF (gamma point) | `pbc.scf.RHF` | |
| KRHF | `pbc.scf.KRHF` | |
| RKS (gamma point) | `pbc.dft.RKS` | |
| KRKS | `pbc.dft.KRKS` | |
| FFTDF | `pbc.df.FFTDF` | the only density fitting scheme |
| Fully jittable KRHF | `pbc.scf.khf_lite.KSCFLite` | used with `pbc.gto.CellLite` |

Unrestricted periodic mean fields, GDF/MDF/RSDF, and the periodic
correlated methods are not implemented.

## Derivatives

Where a method is listed above, first derivatives with respect to the
traced attributes of `gto.Mole` (`coords`, `exp`, `ctr_coeff`) are
supported. Higher order derivatives are covered by the test suite for the
molecular integrals, the SCF and DFT methods, and RCCSD; elsewhere they
may work but are not tested.

```{warning}
Basis parameter derivatives on the legacy `gto.Mole` path are **not**
taken with respect to the raw parameters of the basis set. `exp` and
`ctr_coeff` are read out of the already built `mol._env`, into which pyscf
has folded the primitive and contracted-AO normalization: `ctr_coeff`
holds the normalized contraction coefficients rather than the ones in the
basis set definition, and the `exp` derivative is taken with those
coefficients held fixed, so it misses the dependence of the normalization
factors on the exponent. `gto.MoleLite` instead builds the basis inside
the traced computation, and so differentiates the raw exponents and
coefficients.
```
