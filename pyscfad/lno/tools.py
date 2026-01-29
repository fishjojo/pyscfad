# Copyright 2023-2026 The PySCFAD Authors
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

import numpy as np
from pyscf.lib import logger

def autofrag(mol, H2heavy=True):
    r'''Partition molecule into fragments.

    Args:
        mol : Mole
            Molecular information.
        H2heavy : bool
            If set to ``True``, hydrogen atoms are bound with the nearest heavy atom to
            make a fragment. Otherwise, every atom, including hydrogen, makes a
            fragment.
    '''
    if H2heavy:
        get_dist = lambda x,y: ((x[:,None,:]-y)**2.).sum(axis=-1)

        if hasattr(mol, 'lattice_vectors'):  # mol is actually a Cell object
            alat = mol.lattice_vectors()
        else:
            alat = None
        cs = np.asarray(mol.atom_charges())
        rs = np.asarray(mol.atom_coords())
        idx_H = np.where(cs == 1)[0]
        idx_X = np.where(cs > 1)[0]
        if idx_X.size > 0:
            if alat is None:
                d2 = get_dist(rs[idx_H], rs[idx_X])
                H2Xmap = np.argmin(d2, axis=1)
            else:
                d2 = []
                for jx in [-1,0,1]:
                    for jy in [-1,0,1]:
                        for jz in [-1,0,1]:
                            a = np.dot(np.array([jx,jy,jz]), alat)
                            d2.append( get_dist(rs[idx_H], rs[idx_X]+a) )
                d2 = np.hstack(d2)
                H2Xmap = np.argmin(d2, axis=1) % len(idx_X)
            frag_atmlist = [None] * len(idx_X)
            for i,iX in enumerate(idx_X):
                iHs = np.where(H2Xmap==i)[0]
                l = np.asarray(np.concatenate([[iX], idx_H[iHs]]),
                               dtype=int).tolist()
                frag_atmlist[i] = l
        else:   # all-H system
            print('warning: no heavy atom detected in the system; every '
                  'hydrogen atom is treated as a single fragment.')
            frag_atmlist = [[i] for i in idx_H]
    else:
        frag_atmlist = [[i] for i in np.where(mol.atom_charges() > 0)[0]]

    return frag_atmlist

def _matpow(A, p):
    e, u = np.linalg.eigh(A)
    return np.dot(u*e**p, u.T)

def map_lo_to_frag(mol, orbloc, frag_atmlist, verbose=None):
    r''' Assign input LOs (assumed orthonormal) to fragments using the Lowdin charge.

    For each IAO 'i', a 1D array, [p_1, p_2, ... p_nfrag], is computed, where
        p_ifrag = \sum_{mu on fragment i} ( (s1e^{1/2}*orbloc)[mu,i] )**2.
    '''
    if verbose is None: verbose = mol.verbose
    log = logger.Logger(mol.stdout, verbose)

    if hasattr(mol, 'pbc_intor'):
        s1e = mol.pbc_intor('int1e_ovlp')
    else:
        s1e = mol.intor('int1e_ovlp')
    s1e_sqrt = _matpow(s1e, 0.5)
    plo_ao = np.dot(s1e_sqrt, orbloc) ** 2
    aoslice_by_atom = mol.aoslice_nr_by_atom()
    aoind_by_frag = [np.concatenate([range(*aoslice_by_atom[atm][-2:])
                                     for atm in atmlist])
                     for atmlist in frag_atmlist]
    plo_frag = np.array([plo_ao[aoind].sum(axis=0)
                         for aoind in aoind_by_frag]).T
    lo_frag_map = plo_frag.argmax(axis=1)
    nlo, nfrag = plo_frag.shape
    for i in range(nlo):
        log.debug1('IAO %d is assigned to frag %d with charge %.2f',
                   i, lo_frag_map[i], plo_frag[i,lo_frag_map[i]])
        log.debug2('pop by frag:' + ' %.2f'*nfrag, *plo_frag[i])

    frag_lolist = [np.where(lo_frag_map==i)[0] for i in range(nfrag)]

    return frag_lolist
