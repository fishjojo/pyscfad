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

from pyscf.lib import logger
from pyscfad import config

def intor(mol, intor_name, comp=None, hermi=0, aosym='s1', out=None,
          shls_slice=None, grids=None):
    if config.moleintor_opt:
        from pyscfad.gto._moleintor_vjp import (intor2c, intor3c, intor4c)
    else:
        from pyscfad.gto._moleintor_jvp import (intor2c, intor3c, intor4c)

    if '_spinor' in intor_name:
        msg = 'integrals for spinors are not differentiable.'
        logger.warn(mol, msg)
    if grids is not None:
        msg = 'integrals on grids are not differentiable.'
        logger.warn(mol, msg)
    if out is not None:
        msg = f'argument out = {out} will be ignored with AD.'
        logger.warn(mol, msg)
    if hermi == 2:
        msg = 'integrals with anti-hermitian symmetry are not differentiable.'
        logger.warn(mol, msg)
    #if aosym != 's1':
    #    msg = 'not all AO symmetries are supported for differentiation.'
    #    logger.warn(mol, msg)

    if (intor_name.startswith('int1e') or
        intor_name.startswith('int2c2e') or
        intor_name.startswith('ECP')):
        return intor2c(mol, intor_name, comp, hermi, aosym, out, shls_slice, grids)
    elif intor_name.startswith('int2e') or intor_name.startswith('int4c1e'):
        return intor4c(mol, intor_name, comp, hermi, aosym, out, shls_slice, grids)
    elif intor_name.startswith('int3c'):
        return intor3c(mol, intor_name, comp, hermi, aosym, out, shls_slice, grids)
    else:
        raise KeyError(f'Unknown integral name: {intor_name}.')

def intor_cross(intor_name, mol1, mol2, comp=None, grids=None):
    from pyscfad.gto._moleintor_jvp import intor_cross as _intor_cross
    return _intor_cross(intor_name, mol1, mol2, comp=comp, grids=grids)
