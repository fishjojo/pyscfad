# Copyright 2021-2025 Xing Zhang
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
import tomllib
import numpy
from pyscf.data.nist import HARTREE2EV

from pyscfad import numpy as np
from pyscfad import pytree
from pyscfad.gto.mole import inter_distance
from pyscfad.pbc import gto as pbcgto
from pyscfad.xtb import util
from pyscfad.xtb.data.radii import COV_D3

DefaultParamFile = os.path.join(os.path.dirname(__file__), "data/gfn1-xtb.toml")


def cn_d3(mol, charges=None, coords=None, kcn=16.0, cov_radii=None):
    if charges is None:
        charges = mol.atom_charges()
    if coords is None:
        coords = mol.atom_coords()
    if cov_radii is None:
        cov_radii = COV_D3[charges]

    Ls = None
    if isinstance(mol, pbcgto.Cell):
        Ls = mol.get_lattice_Ls()

    RAB = cov_radii[:,None] + cov_radii[None,:]
    r = inter_distance(coords=coords, Ls=Ls)
    r = np.where(r>1e-6, r, np.inf)
    CN = np.where(r>1e-6, 1. / (1. + np.exp(-kcn * (RAB / r - 1.))), 0)

    if CN.ndim == 2:
        axis = 1
    elif CN.ndim == 3:
        axis = (0, 2)
    else:
        raise ValueError(f"CN has wrong dimension of {CN.ndim}")
    return np.sum(CN, axis=axis)


def load_param(data_or_file: dict | str = DefaultParamFile) -> dict:
    if isinstance(data_or_file, dict):
        data = data_or_file
    elif isinstance(data_or_file, str):
        with open(data_or_file, "rb") as f:
            data = tomllib.load(f)
    return data


class Element(pytree.PytreeNode):
    _dynamic_attr = [
        "levels",
        "slater",
        "refocc",
        "shpoly",
        "kcn",
        "gam",
        "lgam",
        "gam3",
        "zeff",
        "arep",
        "xbond",
        "en",
        "dkernel",
        "qkernel",
        "mprad",
        "mpvcn",
    ]
    def __init__(self, data: dict):
        self.shells = list(data.get("shells"))
        self.levels = numpy.asarray(data.get("levels"))
        self.slater = numpy.asarray(data.get("slater"))
        self.ngauss = list(data.get("ngauss"))
        self.refocc = numpy.asarray(data.get("refocc"))
        self.shpoly = numpy.asarray(data.get("shpoly"))
        self.kcn = numpy.asarray(data.get("kcn"))
        self.gam = numpy.asarray(data.get("gam"))
        self.lgam = numpy.asarray(data.get("lgam"))
        self.gam3 = numpy.asarray(data.get("gam3"))
        self.zeff = numpy.asarray(data.get("zeff"))
        self.arep = numpy.asarray(data.get("arep"))
        self.xbond = numpy.asarray(data.get("xbond"))
        self.en = numpy.asarray(data.get("en"))
        self.dkernel = numpy.asarray(data.get("dkernel"))
        self.qkernel = numpy.asarray(data.get("qkernel"))
        self.mprad = numpy.asarray(data.get("mprad"))
        self.mpvcn = numpy.asarray(data.get("mpvcn"))


class GFN1Param(pytree.PytreeNode):
    """Selected parameters from the GFN1 parameter set
    """
    element = {}
    kpair = None
    k_shlpr = None
    kEN = None
    kcn_d3 = None

    _dynamic_attr = [
        "element",
        "kpair",
        "k_shlpr",
        "kEN",
        "kf",
        "kcn_d3",
    ]
    def __init__(self, data: dict = None):
        if data is None:
            data = load_param()

        for k, v in data["element"].items():
            self.element[k] = Element(v)

        self.kpair = data["hamiltonian"]["xtb"]["kpair"]
        self.k_shlpr = data["hamiltonian"]["xtb"]["shell"]
        self.kEN = data["hamiltonian"]["xtb"]["enscale"]
        self.kf = data["repulsion"]["effective"]["kexp"]
        self.kcn_d3 = data["hamiltonian"]["xtb"]["kcn_d3"]

    def to_mol_param(self, mol):
        return GFN1MolParam(mol, self)


class GFN1MolParam(pytree.PytreeNode):
    """GFN1 parameters for a molecule
    """
    _dynamic_attr = [
        "EN",
        "gam",
        "gam3",
        "zeff",
        "arep",
        "refocc",
        "lgam",
        "shpoly",
        "selfenergy",
        "kcn",
        "kpair",
        "k_shlpr",
        "kf",
        "kEN",
        "CN",
    ]
    def __init__(self, mol, param):
        self.EN   = util.load_unique_element_params(mol, param, "en", broadcast="atom")
        self.gam  = util.load_unique_element_params(mol, param, "gam", broadcast="shell")
        self.gam3 = util.load_unique_element_params(mol, param, "gam3", broadcast="atom")
        self.zeff = util.load_unique_element_params(mol, param, "zeff", broadcast="atom")
        self.arep = util.load_unique_element_params(mol, param, "arep", broadcast="atom")

        self.refocc = util.load_unique_element_shell_params(mol, param, "refocc", broadcast="shell")
        self.lgam   = util.load_unique_element_shell_params(mol, param, "lgam", broadcast="shell")
        self.shpoly = util.load_unique_element_shell_params(mol, param, "shpoly", broadcast="shell")
        self.selfenergy = util.load_unique_element_shell_params(
                                mol, param, "levels", pad=HARTREE2EV, broadcast="shell") / HARTREE2EV
        self.kcn = util.load_unique_element_shell_params(mol, param, "kcn", broadcast="shell") / HARTREE2EV

        self.kpair = util.load_global_element_pair_params(mol, param, "kpair", broadcast="shell")
        self.k_shlpr = util.load_global_shell_pair_params_gfn1(mol, param, "k_shlpr", broadcast="shell")

        self.kf = param.kf
        self.kEN = param.kEN
        self.CN = cn_d3(mol, kcn=param.kcn_d3)


if __name__ == "__main__":
    import jax
    cell = pbcgto.Cell()
    cell.a = np.eye(3) * 3
    cell.atom = """
        O  0.000000  0.000000  0.000000
        H  0.758602  0.000000  0.504284
        H  0.758602  0.000000  -0.504284
    """
    cell.basis = "./basis/gfn1.dat"
    cell.build()

    def foo(param):
        return param.element["H"].gam * param.element["He"].gam * param.kpair["H-H"] * param.kEN * param.kf* param.kcn_d3

    param = GFN1Param()
    g = jax.jit(jax.grad(foo))(param)
    print(g.element["H"].gam, g.element["He"].gam, g.kpair["H-H"], g.kEN, g.kf, g.kcn_d3)

    cell_param = param.to_mol_param(cell)
    print(cell_param.k_shlpr)
