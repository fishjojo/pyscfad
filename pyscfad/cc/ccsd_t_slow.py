'''
CCSD(T)
'''
from pyscf import numpy as np
from pyscf.lib import logger
from pyscfad.lib import jit, vmap

# t3 as ijkabc

# JCP 94, 442 (1991); DOI:10.1063/1.460359.  Error in Eq (1), should be [ia] >= [jb] >= [kc]
def kernel(mycc, eris, t1=None, t2=None, verbose=logger.NOTE):
    log = logger.new_logger(mycc, verbose)

    if t1 is None: t1 = mycc.t1
    if t2 is None: t2 = mycc.t2

    nocc, nvir = t1.shape
    mo_e = eris.mo_energy
    e_occ, e_vir = mo_e[:nocc], mo_e[nocc:]
    #eijk = lib.direct_sum('i,j,k->ijk', e_occ, e_occ, e_occ)
    eijk = e_occ[:,None,None] + e_occ[None,:,None] + e_occ[None,None,:]

    eris_vvov = eris.get_ovvv().conj().transpose(1,3,0,2)
    eris_vooo = np.asarray(eris.ovoo).conj().transpose(1,0,3,2)
    eris_vvoo = np.asarray(eris.ovov).conj().transpose(1,3,0,2)
    fvo = eris.fock[nocc:,:nocc]

    idx = []
    scal = []
    for a in range(nvir):
        for b in range(a+1):
            for c in range(b+1):
                idx.append([a,b,c])
                if a == c:
                    fac = 6
                elif a == b or b == c:
                    fac = 2
                else:
                    fac = 1
                scal.append(fac)

    idx = np.asarray(idx, dtype=np.int32)
    scal = np.asarray(scal, dtype=float)

    et = _compute_et(t1, t2, eris_vvov, eris_vooo, eris_vvoo,
                     fvo, eijk, e_vir, idx, scal)
    log.info('CCSD(T) correction = %.15g', et)
    return et

@jit
def _compute_et(t1, t2, eris_vvov, eris_vooo, eris_vvoo,
                fvo, eijk, e_vir, idx, scal):
    t1T = t1.T
    t2T = t2.transpose(2,3,0,1)

    def get_w(a, b, c):
        w = np.einsum('if,fkj->ijk', eris_vvov[a,b], t2T[c,:])
        w-= np.einsum('ijm,mk->ijk', eris_vooo[a,:], t2T[b,c])
        return w

    def get_v(a, b, c):
        v = np.einsum('ij,k->ijk', eris_vvoo[a,b], t1T[c])
        v+= np.einsum('ij,k->ijk', t2T[a,b], fvo[c])
        return v

    def r3(w):
        return (4 * w + w.transpose(1,2,0) + w.transpose(2,0,1)
                - 2 * w.transpose(2,1,0) - 2 * w.transpose(0,2,1)
                - 2 * w.transpose(1,0,2))

    def body(fac, index):
        a, b, c = index[:]
        d3  = eijk - e_vir[a] - e_vir[b] - e_vir[c]
        d3 *= fac

        wabc = get_w(a, b, c)
        wacb = get_w(a, c, b)
        wbac = get_w(b, a, c)
        wbca = get_w(b, c, a)
        wcab = get_w(c, a, b)
        wcba = get_w(c, b, a)

        WW = (wabc
              + wacb.transpose(0,2,1) + wbac.transpose(1,0,2)
              + wbca.transpose(2,0,1) + wcab.transpose(1,2,0)
              + wcba.transpose(2,1,0))

        vabc = get_v(a, b, c)
        vacb = get_v(a, c, b)
        vbac = get_v(b, a, c)
        vbca = get_v(b, c, a)
        vcab = get_v(c, a, b)
        vcba = get_v(c, b, a)

        VV = (vabc
              + vacb.transpose(0,2,1) + vbac.transpose(1,0,2)
              + vbca.transpose(2,0,1) + vcab.transpose(1,2,0)
              + vcba.transpose(2,1,0))

        ZZ = r3(WW + .5 * VV) / d3

        et = np.einsum('ijk,ijk', WW, ZZ.conj())
        return et

    et = vmap(body, in_axes=(0,0), signature='(),(x)->()')(scal, idx)
    et = np.sum(et) * 2
    return et
