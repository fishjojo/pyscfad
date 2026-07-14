"""GFN1-KXTB band structure of silicon.

Converge the SCF on a Monkhorst-Pack mesh, then diagonalize the converged
Fock operator along a high-symmetry k-path (L - Gamma - X). The
charge-dependent potential is frozen at the converged shell charges, so the
band Hamiltonian at any k is H(k) = Hcore(k) + Veff[q](k).
"""
import numpy
from pyscf.data.nist import BOHR, HARTREE2EV
from pyscfad import numpy as np
from pyscfad.pbc.gto import CellLite
from pyscfad.xtb import basis as xtb_basis
from pyscfad.xtb.param import GFN1Param
from pyscfad.xtb.kxtb import GFN1KXTB

# Si diamond
numbers = [14, 14]
coords = np.array(
    [
        [0.00000, 0.00000, 0.00000],
        [1.3467560987, 1.3467560987, 1.3467560987],
    ]
) / BOHR
a = np.array(
    [
        [0.0, 2.6935121974, 2.6935121974],
        [2.6935121974, 0.0, 2.6935121974],
        [2.6935121974, 2.6935121974, 0.0],
    ]
) / BOHR

basis = xtb_basis.get_basis_filename()
param = GFN1Param()

cell = CellLite(numbers=numbers, coords=coords, a=a, basis=basis,
                precision=1e-6, verbose=0)
kpts = cell.make_kpts([4, 4, 4])

# --- SCF on the Monkhorst-Pack mesh ---
mf = GFN1KXTB(cell, param=param, kpts=kpts)
mf.diis = "anderson"
mf.conv_tol = 1e-10
e_tot = mf.kernel()
print(f"SCF energy: {e_tot:.10f} Ha")

# converged shell charges freeze the second/third-order potential
q = mf.get_q()

# --- high-symmetry path L - Gamma - X (scaled/fractional coordinates) ---
L = numpy.array([0.5, 0.5, 0.5])
G = numpy.array([0.0, 0.0, 0.0])
X = numpy.array([0.5, 0.0, 0.5])

def segment(k0, k1, n):
    return k0 + (k1 - k0) * numpy.linspace(0.0, 1.0, n)[:, None]

scaled_path = numpy.vstack([segment(L, G, 21), segment(G, X, 21)[1:]])
band_kpts = np.asarray(cell.get_abs_kpts(scaled_path))

# --- bands: diagonalize H(k) = Hcore(k) + Veff[q](k) along the path ---
s1e_b = mf.get_ovlp(kpts=band_kpts)
h1e_b = mf.get_hcore(kpts=band_kpts)
vxc_b = mf.get_veff(s1e=s1e_b, kpts=band_kpts, q=q)
fock_b = h1e_b + vxc_b.vxc

mo_energy, _ = mf._eigh(fock_b, s1e_b)
bands = numpy.sort(numpy.asarray(mo_energy).real, axis=1) * HARTREE2EV

nocc = int(numpy.asarray(mf.tot_electrons)) // 2 // len(kpts)
vbm = bands[:, nocc - 1].max()
cbm = bands[:, nocc].min()
print(f"valence bands per k-point: {nocc}")
print(f"VBM = {vbm:.4f} eV, CBM = {cbm:.4f} eV, gap = {cbm - vbm:.4f} eV")

# path length coordinate for plotting
dk = numpy.linalg.norm(numpy.diff(numpy.asarray(band_kpts), axis=0), axis=1)
x = numpy.concatenate([[0.0], numpy.cumsum(dk)])

print("\n  k-path      bands relative to VBM (eV)")
labels = {0: "L", 20: "G", len(scaled_path) - 1: "X"}
for ik in range(0, len(scaled_path), 4):
    tag = labels.get(ik, " ")
    row = " ".join(f"{b - vbm:8.3f}" for b in bands[ik, : nocc + 2])
    print(f"{tag:2s} x={x[ik]:5.2f} {row}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for band in range(bands.shape[1]):
        plt.plot(x, bands[:, band] - vbm, color="C0", lw=1)
    for ik, name in labels.items():
        plt.axvline(x[ik], color="gray", lw=0.5)
        plt.text(x[ik], plt.ylim()[0], name)
    plt.axhline(0.0, color="gray", lw=0.5, ls="--")
    plt.ylabel("E - VBM (eV)")
    plt.title("Si GFN1-xTB band structure (L-G-X)")
    plt.savefig("si_bands.png", dpi=150, bbox_inches="tight")
    print("\nsaved plot to si_bands.png")
except ImportError:
    pass
