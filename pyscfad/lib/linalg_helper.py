import sys
import warnings
import numpy
from jax import scipy
from pyscf import __config__
from pyscf.lib.linalg_helper import (
    LinearDependenceError,
    _sort_by_similarity,
    _sort_elast,
)
from pyscfad import numpy as np
from pyscfad.lib import logger
from pyscfad.ops import stop_grad, jit

DAVIDSON_LINDEP = getattr(__config__, 'lib_linalg_helper_davidson_lindep', 1e-14)
MAX_MEMORY = getattr(__config__, 'lib_linalg_helper_davidson_max_memory', 2000)
SORT_EIG_BY_SIMILARITY = \
    getattr(__config__, 'lib_linalg_helper_davidson_sort_eig_by_similiarity', False)
FOLLOW_STATE = getattr(__config__, 'lib_linalg_helper_davidson_follow_state', False)

# modified from pyscf v2.3

def make_diag_precond(diag, level_shift=0):
    def precond(dx, e, *args):
        diagd = diag - (e - level_shift)
        diagd = diagd.at[abs(diagd)<1e-8].set(1e-8)
        return dx/diagd
    return precond

@jit
def _fill_heff_hermitian(heff, xs, ax, xt, axt):
    nrow = len(axt)
    row1 = len(ax)
    row0 = row1 - nrow
    for ip, i in enumerate(range(row0, row1)):
        for jp, j in enumerate(range(row0, i)):
            #:heff[i,j] = dot(xt[ip].conj(), axt[jp])
            heff = heff.at[i,j].set(np.dot(xt[ip].conj(), axt[jp]))
            #:heff[j,i] = heff[i,j].conj()
            heff = heff.at[j,i].set(heff[i,j].conj())
        #:heff[i,i] = dot(xt[ip].conj(), axt[ip]).real
        heff = heff.at[i,i].set(np.dot(xt[ip].conj(), axt[ip]).real)

    for i in range(row0):
        axi = np.asarray(ax[i])
        for jp, j in enumerate(range(row0, row1)):
            #:heff[j,i] = dot(xt[jp].conj(), axi)
            heff = heff.at[j,i].set(np.dot(xt[jp].conj(), axi))
            #:heff[i,j] = heff[j,i].conj()
            heff = heff.at[i,j].set(heff[j,i].conj())
        axi = None
    return heff

def _qr(xs, dot, lindep=1e-14):
    nvec = len(xs)
    dtype = xs[0].dtype
    qs = np.empty((nvec,xs[0].size), dtype=dtype)
    rmat = np.empty((nvec,nvec), dtype=dtype)

    nv = 0
    for i in range(nvec):
        xi = np.array(xs[i], copy=True)
        rmat = rmat.at[:,nv].set(0)
        rmat = rmat.at[nv,nv].set(1)
        for j in range(nv):
            prod = dot(qs[j].conj(), xi)
            xi -= qs[j] * prod
            rmat = rmat.at[:,nv].add(-rmat[:,j] * prod)
        innerprod = dot(xi.conj(), xi).real
        norm = np.sqrt(innerprod)
        if innerprod > lindep:
            qs = qs.at[nv].set(xi/norm)
            rmat = rmat.at[:nv+1,nv].divide(norm)
            nv += 1
    return qs[:nv], np.linalg.inv(rmat[:nv,:nv])

@jit
def _outprod_to_subspace(v, xs):
    ndim = v.ndim
    if ndim == 1:
        v = v[:,None]
    space, nroots = v.shape
    x0 = np.einsum('c,x->cx', v[space-1], np.asarray(xs[space-1]))
    for i in reversed(range(space-1)):
        xsi = np.asarray(xs[i])
        for k in range(nroots):
            x0 = x0.at[k].add(v[i,k] * xsi)
    if ndim == 1:
        x0 = x0[0]
    return x0
_gen_x0 = _outprod_to_subspace

def _project_xt_(xt, xs, e, threshold, dot, precond):
    ill_precond = False
    for i, xsi in enumerate(xs):
        xsi = np.asarray(xsi)
        for k, xi in enumerate(xt):
            if xi is None:
                continue
            ovlp = dot(xsi.conj(), xi)
            # xs[i] == xt[k]
            if abs(1 - ovlp)**2 < threshold:
                ill_precond = True
                # rebuild xt[k] to remove correlation between xt[k] and xs[i]
                xi = precond(xi, e[k], xi)
                ovlp = dot(xsi.conj(), xi)
            xi -= xsi * ovlp
            xt[k] = xi
        xsi = None
    return xt, ill_precond

def _normalize_xt_(xt, threshold, dot):
    norm_min = 1
    out = []
    for i, xi in enumerate(xt):
        if xi is None:
            continue
        norm = dot(xi.conj(), xi).real ** .5
        if norm**2 > threshold:
            xt[i] = xt[i] * 1/norm
            norm_min = min(norm_min, norm)
            out.append(xt[i])
    return out, norm_min

def davidson(aop, x0, precond, tol=1e-12, max_cycle=50, max_space=12,
             lindep=DAVIDSON_LINDEP, max_memory=MAX_MEMORY,
             dot=np.dot, callback=None,
             nroots=1, lessio=False, pick=None, verbose=logger.WARN,
             follow_state=FOLLOW_STATE):
    e, x = davidson1(lambda xs: [aop(x) for x in xs],
                     x0, precond, tol, max_cycle, max_space, lindep,
                     max_memory, dot, callback, nroots, lessio, pick, verbose,
                     follow_state)[1:]
    if nroots == 1:
        return e[0], x[0]
    else:
        return e, x

def davidson1(aop, x0, precond, tol=1e-12, max_cycle=50, max_space=12,
              lindep=DAVIDSON_LINDEP, max_memory=MAX_MEMORY,
              dot=np.dot, callback=None,
              nroots=1, lessio=False, pick=None, verbose=logger.WARN,
              follow_state=FOLLOW_STATE, tol_residual=None,
              fill_heff=_fill_heff_hermitian):
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(sys.stdout, verbose)

    if tol_residual is None:
        toloose = numpy.sqrt(tol)
    else:
        toloose = tol_residual
    log.debug1('tol %g  toloose %g', tol, toloose)

    if not callable(precond):
        precond = make_diag_precond(precond)

    if callable(x0):  # lazy initialization to reduce memory footprint
        x0 = x0()
    if getattr(x0, 'ndim', None) == 1:
        x0 = [x0]
    #max_cycle = min(max_cycle, x0[0].size)
    max_space = max_space + (nroots-1) * 4
    # max_space*2 for holding ax and xs, nroots*2 for holding axt and xt
    _incore = max_memory*1e6/x0[0].nbytes > max_space*2+nroots*3
    lessio = lessio and not _incore
    log.debug1('max_cycle %d  max_space %d  max_memory %d  incore %s',
               max_cycle, max_space, max_memory, _incore)
    dtype = None
    heff = None
    fresh_start = True
    e = None
    v = None
    conv = numpy.zeros(nroots, dtype=bool)
    emin = None
    level_shift = 0

    for icyc in range(max_cycle):
        if fresh_start:
            if _incore:
                xs = []
                ax = []
            else:
                raise NotImplementedError
            space = 0
# Orthogonalize xt space because the basis of subspace xs must be orthogonal
# but the eigenvectors x0 might not be strictly orthogonal
            xt = None
            x0len = len(x0)
            xt = _qr(x0, dot, lindep)[0]
            if len(xt) != x0len:
                log.warn('QR decomposition removed %d vectors.', x0len - len(xt))
                if callable(pick):
                    log.warn('Check to see if `pick` function %s is providing '
                             'linear dependent vectors', pick.__name__)
                if len(xt) == 0:
                    if icyc == 0:
                        msg = 'Initial guess is empty or zero'
                    else:
                        msg = ('No more linearly independent basis were found. '
                               'Unless loosen the lindep tolerance (current value '
                               f'{lindep}), the diagonalization solver is not able '
                               'to find eigenvectors.')
                    raise LinearDependenceError(msg)
            x0 = None
            max_dx_last = 1e9
            if SORT_EIG_BY_SIMILARITY:
                conv = numpy.zeros(nroots, dtype=bool)
        elif len(xt) > 1:
            xt = _qr(xt, dot, lindep)[0]
            xt = xt[:40]  # 40 trial vectors at most

        axt = aop(xt)
        for k, xi in enumerate(xt):
            xs.append(xi)
            ax.append(axt[k])
        rnow = len(xt)
        #head, space = space, space+rnow
        space = space+rnow

        if dtype is None:
            try:
                dtype = numpy.result_type(axt[0], xt[0])
            except IndexError as exc:
                raise LinearDependenceError('No linearly independent basis found '
                                            'by the diagonalization solver.') from exc
        if heff is None:  # Lazy initilize heff to determine the dtype
            heff = np.empty((max_space+nroots,max_space+nroots), dtype=dtype)
        else:
            heff = np.asarray(heff, dtype=dtype)
        elast = e
        vlast = v
        conv_last = conv

        heff = fill_heff(heff, xs, ax, xt, axt)
        xt = axt = None
        w, v = scipy.linalg.eigh(heff[:space,:space])
        if callable(pick):
            w, v, _ = pick(w, v, nroots, locals())
            if len(w) == 0:
                raise RuntimeError(f'Not enough eigenvalues found by {pick}')

        if SORT_EIG_BY_SIMILARITY:
            e, v = _sort_by_similarity(w, v, nroots, conv, stop_grad(vlast), emin)
        else:
            e = w[:nroots]
            v = v[:,:nroots]
            conv = numpy.zeros(nroots, dtype=bool)
            elast, conv_last = _sort_elast(elast, conv_last,
                                           stop_grad(vlast),
                                           stop_grad(v),
                                           fresh_start, log)

        if elast is None:
            de = e
        elif elast.size != e.size:
            log.debug('Number of roots different from the previous step (%d,%d)',
                      e.size, elast.size)
            de = e
        else:
            de = e - elast
        x0 = None
        x0 = _gen_x0(v, xs)
        if lessio:
            ax0 = aop(x0)
        else:
            ax0 = _gen_x0(v, ax)

        dx_norm = numpy.zeros(nroots)
        xt = [None] * nroots
        for k, ek in enumerate(e):
            if not conv[k]:
                xt[k] = ax0[k] - ek * x0[k]
                dx_norm[k] = numpy.sqrt(numpy.dot(stop_grad(xt[k].conj()), stop_grad(xt[k])).real)
                conv[k] = abs(de[k]) < tol and dx_norm[k] < toloose
                if conv[k] and not conv_last[k]:
                    log.debug('root %d converged  |r|= %4.3g  e= %s  max|de|= %4.3g',
                              k, dx_norm[k], ek, de[k])
        ax0 = None
        max_dx_norm = max(dx_norm)
        ide = numpy.argmax(abs(stop_grad(de)))
        if all(conv):
            log.debug('converged %d %d  |r|= %4.3g  e= %s  max|de|= %4.3g',
                      icyc, space, max_dx_norm, e, de[ide])
            break
        elif (follow_state and max_dx_norm > 1 and
              max_dx_norm/max_dx_last > 3 and space > nroots+2):
            log.debug('davidson %d %d  |r|= %4.3g  e= %s  max|de|= %4.3g',
                      icyc, space, max_dx_norm, e, de[ide])
            log.debug('Large |r| detected, restore to previous x0')
            x0 = _gen_x0(vlast, xs)
            fresh_start = True
            continue
        if SORT_EIG_BY_SIMILARITY:
            if any(conv) and e.dtype == np.double:
                emin = min(e)

        # remove subspace linear dependency
        for k, ek in enumerate(e):
            if (not conv[k]) and dx_norm[k]**2 > lindep:
                xt[k] = precond(xt[k], e[0]-level_shift, x0[k])
                xt[k] *= dot(xt[k].conj(), xt[k]).real ** -.5
            elif not conv[k]:
                # Remove linearly dependent vector
                xt[k] = None
                log.debug1('Drop eigenvector %d, norm=%4.3g', k, dx_norm[k])
            else:
                xt[k] = None

        xt, ill_precond = _project_xt_(xt, xs, e, lindep, dot, precond)
        if ill_precond:
            # Manually adjust the precond because precond function may not be
            # able to generate linearly dependent basis vectors. e.g. issue 1362
            log.warn('Matrix may be already a diagonal matrix. '
                     'level_shift is applied to precond')
            level_shift = 0.1

        xt, norm_min = _normalize_xt_(xt, lindep, dot)
        log.debug('davidson %d %d  |r|= %4.3g  e= %s  max|de|= %4.3g  lindep= %4.3g',
                  icyc, space, max_dx_norm, e, de[ide], norm_min)
        if len(xt) == 0:
            log.debug('Linear dependency in trial subspace. |r| for each state %s',
                      dx_norm)
            conv[dx_norm < toloose] = True
            break
        max_dx_last = max_dx_norm
        fresh_start = space+nroots > max_space

        if callable(callback):
            callback(locals())

    # pylint: disable=unnecessary-comprehension
    x0 = [x for x in x0]  # nparray -> list

    # Check whether the solver finds enough eigenvectors.
    h_dim = x0[0].size
    if len(x0) < min(h_dim, nroots):
        # Two possible reasons:
        # 1. All the initial guess are the eigenvectors. No more trial vectors
        # can be generated.
        # 2. The initial guess sits in the subspace which is smaller than the
        # required number of roots.
        msg = f'Not enough eigenvectors (len(x0)={len(x0)}, nroots={nroots})'
        warnings.warn(msg)

    return numpy.asarray(conv), e, x0
