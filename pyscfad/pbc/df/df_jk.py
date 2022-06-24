import numpy
from pyscf.pbc import tools as pyscf_tools
from pyscf.pbc.lib.kpts_helper import is_zero, member
from pyscfad.lib import numpy as np
from pyscfad.lib import ops

def _ewald_exxdiv_for_G0(cell, kpts, dms, vk, kpts_band=None):
    s = cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts)
    madelung = pyscf_tools.pbc.madelung(cell, kpts)
    if kpts is None:
        for i,dm in enumerate(dms):
            #vk[i] += madelung * reduce(numpy.dot, (s, dm, s))
            vk = ops.index_add(vk, ops.index[i],
                               madelung * np.dot(s, np.dot(dm, s)))
    elif numpy.shape(kpts) == (3,):
        if kpts_band is None or is_zero(kpts_band-kpts):
            for i,dm in enumerate(dms):
                #vk[i] += madelung * reduce(numpy.dot, (s, dm, s))
                vk = ops.index_add(vk, ops.index[i],
                                   madelung * np.dot(s, np.dot(dm, s)))
    elif kpts_band is None or numpy.array_equal(kpts, kpts_band):
        for k in range(len(kpts)):
            for i,dm in enumerate(dms):
                #vk[i,k] += madelung * reduce(numpy.dot, (s[k], dm[k], s[k]))
                vk = ops.index_add(vk, ops.index[i,k],
                                   madelung * np.dot(s[k], np.dot(dm[k], s[k])))
    else:
        for k, kpt in enumerate(kpts):
            for kp in member(kpt, kpts_band.reshape(-1,3)):
                for i,dm in enumerate(dms):
                    #vk[i,kp] += madelung * reduce(numpy.dot, (s[k], dm[k], s[k]))
                    vk = ops.index_add(vk, ops.index[i,kp],
                                   madelung * np.dot(s[k], np.dot(dm[k], s[k])))
    return vk
