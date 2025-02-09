'''
CCSD(T)
'''
from pyscfad import numpy as np
from pyscfad import config, config_update
from pyscfad import lib
from pyscfad.lib import logger
from pyscfad.ops import jit, vmap
from pyscfad.implicit_diff import make_implicit_diff
from pyscfad.tools.linear_solver import gen_gmres

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


# iterative solver

def get_ovvv(eris, *slices):
    ovw = np.asarray(eris.ovvv[slices])
    nocc, nvir, nvir_pair = ovw.shape
    with config_update('pyscfad_moleintor_opt', False):
        ovvv = lib.unpack_tril(ovw.reshape(nocc*nvir,nvir_pair))
    nvir1 = ovvv.shape[2]
    return ovvv.reshape(nocc,nvir,nvir1,nvir1)

@jit
def get_w_6d(eris, t2):
    eris_vvov = get_ovvv(eris).conj().transpose(1,3,0,2)
    eris_vooo = np.asarray(eris.ovoo).conj().transpose(1,0,3,2)
    t2T = t2.transpose(2,3,0,1)

    w = np.einsum('abif,cfkj->ijkabc', eris_vvov, t2T)
    w-= np.einsum('aijm,bcmk->ijkabc', eris_vooo, t2T)
    return w

@jit
def get_v_6d(eris, t1, t2):
    nocc = t1.shape[0]
    eris_vvoo = np.asarray(eris.ovov).conj().transpose(1,3,0,2)
    fvo = eris.fock[nocc:,:nocc]
    t1T = t1.T
    t2T = t2.transpose(2,3,0,1)

    v = np.einsum('abij,ck->ijkabc', eris_vvoo, t1T)
    v+= np.einsum('abij,ck->ijkabc', t2T, fvo)
    return .5 * v

@jit
def r3_6d(w):
    return (4 * w + w.transpose(2,0,1,3,4,5) + w.transpose(1,2,0,3,4,5)
            - 2 * w.transpose(2,1,0,3,4,5) - 2 * w.transpose(0,2,1,3,4,5)
            - 2 * w.transpose(1,0,2,3,4,5))

@jit
def get_d3(eia):
    d3 = (eia[:,None,None,:,None,None] +
          eia[None,:,None,None,:,None] +
          eia[None,None,:,None,None,:])
    return d3

@jit
def P3(w):
    w = w + w.transpose(1,0,2,4,3,5) + w.transpose(2,1,0,5,4,3)
    return w

@jit
def P6(w):
    w = (w
        +w.transpose(1,0,2,4,3,5)
        +w.transpose(2,1,0,5,4,3)
        +w.transpose(0,2,1,3,5,4)
        +w.transpose(1,2,0,4,5,3)
        +w.transpose(2,0,1,5,3,4))
    return w

def update_amps(t1, t2, t3, eris, initial_guess=False):
    t3new  = get_w_6d(eris, t2)
    t3new += get_v_6d(eris, t1, t2)
    t3new = P6(t3new)

    nocc, nvir = t1.shape
    mo_e = eris.mo_energy
    eia = mo_e[:nocc,None] - mo_e[nocc:]
    d3 = get_d3(eia)

    if not initial_guess:
        fvv = eris.fock[nocc:,nocc:]
        fvv = fvv.at[np.diag_indices(nvir)].set(0)
        tvv = np.einsum('ae,ijkebc->ijkabc', fvv, t3)
        t3new += P3(tvv)

        foo = eris.fock[:nocc,:nocc]
        foo = foo.at[np.diag_indices(nocc)].set(0)
        too = np.einsum('im,mjkabc->ijkabc', foo, t3)
        t3new -= P3(too)
    return t3new / d3

def energy(t2, t3, eris):
    w = get_w_6d(eris, t2)
    w = P6(w)

    nocc = t3.shape[0]
    mo_e = eris.mo_energy
    eia = mo_e[:nocc,None] - mo_e[nocc:]
    d3 = get_d3(eia)
    w /= d3
    et = np.einsum('ijkabc,ijkabc', t3, r3_6d(w)*d3) / 3.
    return et

def run_diis(mycc, t3, istep, normt, de, adiis):
    if (adiis and
        istep >= mycc.diis_start_cycle and
        abs(de) < mycc.diis_start_energy_diff):
        t3 = adiis.update(t3.ravel()).reshape(t3.shape)
    return t3

def _converged_iter(t3, t1, t2, eris):
    t3 = update_amps(t1, t2, t3, eris)
    return t3

def _iter(t3, t1, t2, eris, *, mycc=None,
          diis=None, max_cycle=50, tol=1e-8,
          tolnormt=1e-6, verbose=None):
    log = logger.new_logger(mycc, verbose)

    eold = 0
    et = energy(t2, t3, eris)
    log.info('Init E(T) = %.15g', et)
    cput1 = log.timer('initialize (T)')

    conv = False
    for istep in range(max_cycle):
        t3new = update_amps(t1, t2, t3, eris)
        tmpvec = (t3new - t3).ravel()
        normt = np.linalg.norm(tmpvec)
        tmpvec = None
        t3 = t3new
        t3 = run_diis(mycc, t3, istep, normt, et-eold, diis)
        eold, et = et, energy(t2, t3, eris)
        log.info('cycle = %d  E(T) = %.15g  dE = %.9g  norm(t1,t2) = %.6g',
                 istep+1, et, et - eold, normt)
        cput1 = log.timer('(T) iter', *cput1)
        if abs(et-eold) < tol and normt < tolnormt:
            conv = True
            break

    del log
    return t3, conv

def iterative_kernel(mycc, eris=None, t1=None, t2=None, t3=None,
                     max_cycle=50, tol=1e-8,
                     tolnormt=1e-6, verbose=None):
    log = logger.new_logger(mycc, verbose)
    if eris is None:
        eris = mycc.ao2mo(mycc.mo_coeff)
    if t1 is None:
        t1 = mycc.t1
    if t2 is None:
        t2 = mycc.t2
    if t3 is None:
        t3 = update_amps(t1, t2, t3, eris, initial_guess=True)

    if isinstance(mycc.diis, lib.diis.DIIS):
        adiis = mycc.diis
    elif mycc.diis:
        adiis = lib.diis.DIIS(mycc, mycc.diis_file, incore=mycc.incore_complete)
        adiis.space = mycc.diis_space
    else:
        adiis = None

    t3, conv = make_implicit_diff(_iter, config.ccsd_implicit_diff,
                                   optimality_cond=_converged_iter,
                                   solver=gen_gmres(), has_aux=True)(
                                        t3, t1, t2, eris, mycc=mycc,
                                        diis=adiis, max_cycle=max_cycle, tol=tol,
                                        tolnormt=tolnormt, verbose=log)

    et = energy(t2, t3, eris)
    log.timer('iterative (T)')
    del adiis, log
    return conv, et, t3
