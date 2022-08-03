from pyscf import numpy as np
from pyscf.fci import cistring
from pyscf.fci import fci_slow as pyscf_fci_slow
#from pyscfad.lib import numpy as np
from pyscfad.lib import vmap
from pyscfad.gto import mole

def get_occ_loc(strs, norb):
    locs = []
    for x in strs:
        loc = []
        for i in range(norb):
            if (x >> i) & 1 == 1:
                loc.append(i)
        locs.append(loc)
    return np.asarray(locs)

def fci_ovlp(mol1, mol2, fcivec1, fcivec2, norb1, norb2, nelec1, nelec2, mo1, mo2):
    mo1 = np.asarray(mo1)
    mo2 = np.asarray(mo2)
    fcivec1 = np.asarray(fcivec1)
    fcivec2 = np.asarray(fcivec2)
    if isinstance(nelec1, (int, np.integer)):
        nelecb1 = nelec1//2
        neleca1 = nelec1 - nelecb1
    else:
        neleca1, nelecb1 = nelec1

    if isinstance(nelec2, (int, np.integer)):
        nelecb2 = nelec2//2
        neleca2 = nelec2 - nelecb2
    else:
        neleca2, nelecb2 = nelec2

    sao = mole.intor_cross('int1e_ovlp', mol1, mol2)

    strs_a1 = cistring.make_strings(range(norb1), neleca1)
    locs_a1 = get_occ_loc(strs_a1, norb1)
    if neleca1 == nelecb1:
        locs_b1 = locs_a1
    else:
        strs_b1 = cistring.make_strings(range(norb1), nelecb1)
        locs_b1 = get_occ_loc(strs_b1, norb1)

    if norb1 == norb2 and neleca1 == neleca2:
        locs_a2 = locs_a1
    else:
        strs_a2 = cistring.make_strings(range(norb2), neleca2)
        locs_a2 = get_occ_loc(strs_a2, norb2)
    if norb1 == norb2 and nelecb1 == nelecb2:
        locs_b2 = locs_b1
    else:
        strs_b2 = cistring.make_strings(range(norb2), nelecb2)
        locs_b2 = get_occ_loc(strs_b2, norb2)

    na1 = len(locs_a1)
    nb1 = len(locs_b1)
    na2 = len(locs_a2)
    nb2 = len(locs_b2)

    if getattr(mo1, 'ndim', None) == 2:
        mo_a1 = mo_b1 = mo1
    else:
        mo_a1 = mo1[0]
        mo_b1 = mo1[1]
    if getattr(mo2, 'ndim', None) == 2:
        mo_a2 = mo_b2 = mo2
    else:
        mo_a2 = mo2[0]
        mo_b2 = mo2[1]

    ci1 = fcivec1.reshape(na1,nb1)
    ci2 = fcivec2.reshape(na2,nb2)

    idxa = np.broadcast_to(locs_a2[:,None,:], (na2,nb2,neleca2)).reshape(-1,neleca2)
    idxb = np.broadcast_to(locs_b2[None,:,:], (na2,nb2,nelecb2)).reshape(-1,nelecb2)
    def body(mo_ia, mo_ib, ida, idb):
        sij_a = np.einsum('ui,uv,vj->ij', mo_ia, sao, mo_a2[:,ida])
        sij_b = np.einsum('ui,uv,vj->ij', mo_ib, sao, mo_b2[:,idb])
        val = np.linalg.det(sij_a) * np.linalg.det(sij_b)
        return val

    res = 0.
    for ia in range(na1):
        mo_ia = mo_a1[:,locs_a1[ia]]
        for ib in range(nb1):
            mo_ib = mo_b1[:,locs_b1[ib]]
            val = vmap(body, (None,None,0,0), signature='(i),(j)->()')(mo_ia, mo_ib, idxa, idxb)
            #val = []
            #for i in range(len(idxa)):
            #    val.append(body(mo_ia, mo_ib, idxa[i], idxb[i]))
            #val = np.asarray(val)
            res += ci1[ia,ib] * (val * ci2.ravel()).sum()
    return res


kernel = pyscf_fci_slow.kernel
