# Method coverage

pyscfad re-implements a subset of pyscf's methods so that they can be
differentiated. This page lists what that subset currently is, so that a
missing method can be told apart from a bug.

The tables below are a snapshot of pyscfad 0.3.3. Anything not listed
either has no differentiable counterpart in pyscfad, or is only usable
through `mol.to_pyscf()` / `mf.to_pyscf()`, which strips the traced
attributes and returns a plain pyscf object.

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
