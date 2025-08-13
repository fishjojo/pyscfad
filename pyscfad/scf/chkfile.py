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

import h5py
from pyscf.lib.chkfile import save_mol
from pyscfad.ops import stop_grad
from pyscfad.lib.chkfile import save

def dump_scf(mol, chkfile, e_tot, mo_energy, mo_coeff, mo_occ,
             overwrite_mol=True):
    if h5py.is_hdf5(chkfile) and not overwrite_mol:
        with h5py.File(chkfile, 'a') as fh5:
            if 'mol' not in fh5:
                fh5['mol'] = mol.dumps()
    else:
        save_mol(mol, chkfile)

    scf_dic = {'e_tot'    : stop_grad(e_tot),
               'mo_energy': stop_grad(mo_energy),
               'mo_occ'   : stop_grad(mo_occ),
               'mo_coeff' : stop_grad(mo_coeff)}
    save(chkfile, 'scf', scf_dic)
