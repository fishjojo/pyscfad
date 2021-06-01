import sys
from typing import Any, Optional
import numpy
from pyscf import __config__
from pyscf.pbc.scf import hf as pyscf_pbc_hf
from pyscfad import lib
from pyscfad.scf import hf as mol_hf
from pyscfad.pbc.gto import Cell
from pyscfad.pbc import df

@lib.dataclass
class SCF(mol_hf.SCF, pyscf_pbc_hf.SCF):
    # NOTE every field must have a default value
    cell: Cell = lib.field(pytree_node=True, default=None)
    kpt: numpy.ndarray = numpy.zeros(3)
    with_df: Any = lib.field(pytree_node=True, default=None)
    exxdiv: str = getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald')
    rsjk: Any = None
    conv_tol: Optional[float] = None

    def __init__(self, cell, **kwargs):
        if not cell._built:
            sys.stderr.write('Warning: cell.build() is not called in input\n')
            cell.build()
        self.cell = cell
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])
        if not self._built:
            mol_hf.SCF.__init__(self, cell)

        if self.with_df is None:
            self.with_df = df.FFTDF(self.cell)
        if self.conv_tol is None:
            self.conv_tol = self.cell.precision * 10
        self._keys = self._keys.union(['cell', 'exxdiv', 'with_df'])

    def get_hcore(self, cell=None, kpt=None):
        if cell is None: cell = self.cell
        if kpt is None: kpt = self.kpt
        if cell.pseudo:
            nuc = self.with_df.get_pp(kpt, cell=cell)
        else:
            raise NotImplementedError
            #nuc = self.with_df.get_nuc(kpt)
        if len(cell._ecpbas) > 0:
            raise NotImplementedError
            #nuc += ecp.ecp_int(cell, kpt)
        return nuc + cell.pbc_intor('int1e_kin', 1, 1, kpt)


RHF = SCF
