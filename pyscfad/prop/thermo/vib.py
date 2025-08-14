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

import numpy
from pyscf.hessian import thermo
from pyscf.data.nist import BOHR

def harmonic_analysis(mol, hess, ir_tensor=None, raman_tensor=None,
                      exclude_trans=True, exclude_rot=True,
                      imaginary_freq=True):
    '''
    Computes harmonic vibrational frequency, normal modes,
    IR intensity, Raman activity, Depolarization ratio.

    Args:
        hess: (natm,3,natm,3) array
            The 2nd order derivative of energy w.r.t. atomic coordinates
        ir_tensor: (3,natm,3) array
            The derivative of dipole moment w.r.t. atomic coordinates
        raman_tensor: (3,3,natm,3) array
            The derivative of polarizability w.r.t. atomic coordinates
        exclude_trans: bool
            Whether to exclude translation modes
        exclude_rot: bool
            Whether to exclude rotatoin modes
        imaginary_freq: bool
            Whether to allow imaginary frequency

    Returns:
        vibration: dict
            Results for vibrational analysis
        ir: dict
            Results for IR
        raman: dict
            Results for Raman
    '''

    vibration = thermo.harmonic_analysis(mol, hess.transpose(0,2,1,3),
                                   exclude_trans, exclude_rot, imaginary_freq)
    ir = compute_ir(ir_tensor, vibration)
    raman = compute_raman(raman_tensor, vibration)
    return vibration, ir, raman

def compute_ir(ir_tensor, vibration):
    ir = {'intensity': None} # km/mol
    if ir_tensor is None:
        return ir

    unit_kmmol = 974.8801118351438
    norm_mode = vibration['norm_mode']
    a = numpy.einsum('inx,knx->ki', ir_tensor, norm_mode)
    ir['intensity'] = numpy.einsum('ki,ki->k', a, a) * unit_kmmol
    return ir

def compute_raman(raman_tensor, vibration):
    raman = {'activity': None, # A^4/amu
             'depolar_ratio': None}
    if raman_tensor is None:
        return raman

    norm_mode = vibration['norm_mode']
    chi = raman_tensor * BOHR**2
    alpha = numpy.einsum('ijnx,knx->kij', chi, norm_mode)
    alpha2 = (1./3. * numpy.trace(alpha, axis1=1, axis2=2)) ** 2

    alpha_diag = numpy.diagonal(alpha, axis1=1, axis2=2)
    alpha_ij = alpha_diag[:,:,None] - alpha_diag[:,None,:]
    gamma2  = numpy.einsum('kij->k', 1./4. * alpha_ij ** 2)
    gamma2 += numpy.einsum('kij->k', 1.5 * alpha ** 2)
    gamma2 -= numpy.einsum('kii->k', 1.5 * alpha ** 2)

    raman['activity'] = 45 * alpha2 + 7 * gamma2
    raman['depolar_ratio'] = 3 * gamma2 / (45 * alpha2 + 4 * gamma2)
    return raman
