import numpy
from pyscf.fci import cistring
from pyscfad import numpy as np
from pyscfad import ops
from pyscfad.ops import vmap, stop_grad
from pyscfad.lib.linalg_helper import davidson
from pyscfad.gto import mole
from pyscfad import ao2mo

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

def contract_2e(eri, fcivec, norb, nelec, opt=None):
    '''Compute E_{pq}E_{rs}|CI>'''
    if isinstance(nelec, (int, np.integer)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
    link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    ci0 = fcivec.reshape(na,nb)
    t1 = np.zeros((norb,norb,na,nb))
    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            t1 = ops.index_add(t1, ops.index[a,i,str1], sign * ci0[str0])
    for str0, tab in enumerate(link_indexb):
        for a, i, str1, sign in tab:
            t1 = ops.index_add(t1, ops.index[a,i,:,str1], sign * ci0[:,str0])

    t1 = np.einsum('bjai,aiAB->bjAB', eri.reshape([norb]*4), t1)

    fcinew = np.zeros_like(ci0)
    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            fcinew = ops.index_add(fcinew, ops.index[str1], sign * t1[a,i,str0])
    for str0, tab in enumerate(link_indexb):
        for a, i, str1, sign in tab:
            fcinew = ops.index_add(fcinew, ops.index[:,str1], sign * t1[a,i,:,str0])
    return fcinew.reshape(fcivec.shape)


def absorb_h1e(h1e, eri, norb, nelec, fac=1):
    if not isinstance(nelec, (int, np.integer)):
        nelec = sum(nelec)
    if eri.size != norb**4:
        h2e = ao2mo.restore(1, eri.copy(), norb)
    else:
        h2e = eri.copy().reshape(norb,norb,norb,norb)
    f1e = h1e - np.einsum('jiik->jk', h2e) * .5
    f1e = f1e * (1./(nelec+1e-100))
    for k in range(norb):
        h2e = ops.index_add(h2e, ops.index[k,k,:,:], f1e)
        h2e = ops.index_add(h2e, ops.index[:,:,k,k], f1e)
    return h2e * fac

def make_hdiag(h1e, eri, norb, nelec, opt=None):
    if isinstance(nelec, (int, np.integer)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec

    occslista = cistring.gen_occslst(range(norb), neleca)
    occslistb = cistring.gen_occslst(range(norb), nelecb)
    if eri.size != norb**4:
        eri = ao2mo.restore(1, eri, norb)
    else:
        eri = eri.reshape(norb,norb,norb,norb)
    diagj = np.einsum('iijj->ij', eri)
    diagk = np.einsum('ijji->ij', eri)
    hdiag = []
    for aocc in occslista:
        for bocc in occslistb:
            e1 = h1e[aocc,aocc].sum() + h1e[bocc,bocc].sum()
            e2 = diagj[aocc][:,aocc].sum() + diagj[aocc][:,bocc].sum() \
               + diagj[bocc][:,aocc].sum() + diagj[bocc][:,bocc].sum() \
               - diagk[aocc][:,aocc].sum() - diagk[bocc][:,bocc].sum()
            hdiag.append(e1 + e2*.5)
    return np.array(hdiag)

def kernel(h1e, eri, norb, nelec, ecore=0, nroots=1):
    h2e = absorb_h1e(h1e, eri, norb, nelec, .5)
    na = cistring.num_strings(norb, nelec//2)

    hdiag = make_hdiag(h1e, eri, norb, nelec)
    try:
        from pyscf.fci.direct_spin1 import pspace
        addrs, h0 = pspace(stop_grad(h1e), stop_grad(eri),
                           norb, nelec, stop_grad(hdiag), nroots)
    # pylint: disable=bare-except
    except:
        addrs = numpy.argsort(hdiag)[:nroots]
    ci0 = []
    for addr in addrs:
        x = numpy.zeros((na*na))
        x[addr] = 1.
        ci0.append(x.ravel())

    def hop(c):
        hc = contract_2e(h2e, c, norb, nelec)
        return hc.ravel()
    # pylint: disable=unnecessary-lambda-assignment
    precond = lambda x, e, *args: x/(hdiag-e+1e-4)
    e, c = davidson(hop, ci0, precond, nroots=nroots)
    return e+ecore, c
