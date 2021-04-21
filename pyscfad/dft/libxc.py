import warnings
import math
import numpy
import ctypes
from functools import partial
from jax import custom_jvp
from pyscf.dft.libxc import PROBLEMATIC_XC
from pyscf.dft.libxc import parse_xc, needs_laplacian, _itrf, is_lda, is_meta_gga
from pyscfad.lib import numpy as jnp

def eval_xc(xc_code, rho, spin=0, relativity=0, deriv=1, omega=None, verbose=None):
    hyb, fn_facs = parse_xc(xc_code)
    if omega is not None:
        hyb[2] = float(omega)
    return _eval_xc(rho, hyb, fn_facs, spin, relativity, deriv, verbose)

@partial(custom_jvp, nondiff_argnums=tuple(range(1,7)))
def _eval_xc(rho, hyb, fn_facs, spin=0, relativity=0, deriv=1, verbose=None):
    assert(deriv <= 3)
    if spin == 0:
        nspin = 1
        rho_u = rho_d = numpy.asarray(rho, order='C')
    else:
        nspin = 2
        rho_u = numpy.asarray(rho[0], order='C')
        rho_d = numpy.asarray(rho[1], order='C')
    assert(rho_u.dtype == numpy.double)
    assert(rho_d.dtype == numpy.double)

    if rho_u.ndim == 1:
        rho_u = rho_u.reshape(1,-1)
        rho_d = rho_d.reshape(1,-1)
    ngrids = rho_u.shape[1]

    fn_ids = [x[0] for x in fn_facs]
    facs   = [x[1] for x in fn_facs]
    if hyb[2] != 0:
        # Current implementation does not support different omegas for
        # different RSH functionals if there are multiple RSHs
        omega = [hyb[2]] * len(facs)
    else:
        omega = [0] * len(facs)
    fn_ids_set = set(fn_ids)
    if fn_ids_set.intersection(PROBLEMATIC_XC):
        problem_xc = [PROBLEMATIC_XC[k]
                      for k in fn_ids_set.intersection(PROBLEMATIC_XC)]
        warnings.warn('Libxc functionals %s may have discrepancy to xcfun '
                      'library.\n' % problem_xc)

    if any([needs_laplacian(fid) for fid in fn_ids]):
        raise NotImplementedError('laplacian in meta-GGA method')

    n = len(fn_ids)
    if (n == 0 or  # xc_code = '' or xc_code = 'HF', an empty functional
        all((is_lda(x) for x in fn_ids))):
        if spin == 0:
            nvar = 1
        else:
            nvar = 2
    elif any((is_meta_gga(x) for x in fn_ids)):
        if spin == 0:
            nvar = 4
        else:
            nvar = 9
    else:  # GGA
        if spin == 0:
            nvar = 2
        else:
            nvar = 5
    outlen = (math.factorial(nvar+deriv) //
              (math.factorial(nvar) * math.factorial(deriv)))
    outbuf = numpy.zeros((outlen,ngrids))

    _itrf.LIBXC_eval_xc(ctypes.c_int(n),
                        (ctypes.c_int*n)(*fn_ids),
                        (ctypes.c_double*n)(*facs),
                        (ctypes.c_double*n)(*omega),
                        ctypes.c_int(nspin),
                        ctypes.c_int(deriv), ctypes.c_int(rho_u.shape[1]),
                        rho_u.ctypes.data_as(ctypes.c_void_p),
                        rho_d.ctypes.data_as(ctypes.c_void_p),
                        outbuf.ctypes.data_as(ctypes.c_void_p))

    exc = outbuf[0]
    vxc = fxc = kxc = None
    if nvar == 1:  # LDA
        if deriv > 0:
            vxc = (outbuf[1], None, None, None)
        if deriv > 1:
            fxc = (outbuf[2],) + (None,)*9
        if deriv > 2:
            kxc = (outbuf[3], None, None, None)
    elif nvar == 2:
        if spin == 0:  # GGA
            if deriv > 0:
                vxc = (outbuf[1], outbuf[2], None, None)
            if deriv > 1:
                fxc = (outbuf[3], outbuf[4], outbuf[5],) + (None,)*7
            if deriv > 2:
                kxc = outbuf[6:10]
        else:  # LDA
            if deriv > 0:
                vxc = (outbuf[1:3].T, None, None, None)
            if deriv > 1:
                fxc = (outbuf[3:6].T,) + (None,)*9
            if deriv > 2:
                kxc = (outbuf[6:10].T, None, None, None)
    elif nvar == 5:  # GGA
        if deriv > 0:
            vxc = (outbuf[1:3].T, outbuf[3:6].T, None, None)
        if deriv > 1:
            fxc = (outbuf[6:9].T, outbuf[9:15].T, outbuf[15:21].T) + (None,)*7
        if deriv > 2:
            kxc = (outbuf[21:25].T, outbuf[25:34].T, outbuf[34:46].T, outbuf[46:56].T)
    elif nvar == 4:  # MGGA
        if deriv > 0:
            vxc = outbuf[1:5]
        if deriv > 1:
            fxc = outbuf[5:15]
        if deriv > 2:
            kxc = outbuf[15:19]
    elif nvar == 9:  # MGGA
        if deriv > 0:
            vxc = (outbuf[1:3].T, outbuf[3:6].T, outbuf[6:8].T, outbuf[8:10].T)
        if deriv > 1:
            fxc = (outbuf[10:13].T, outbuf[13:19].T, outbuf[19:25].T,
                   outbuf[25:28].T, outbuf[28:31].T, outbuf[31:35].T,
                   outbuf[35:39].T, outbuf[39:43].T, outbuf[43:49].T,
                   outbuf[49:55].T)
    return exc, vxc, fxc, kxc


@_eval_xc.defjvp
def _eval_xc_jvp(hyb, fn_facs, spin, relativity, deriv, verbose,
                 primals, tangents):
    rho, = primals
    rho_t, = tangents

    assert deriv == 1
    exc, vxc, fxc, kxc = _eval_xc(rho, hyb, fn_facs, spin, relativity, deriv+1, verbose)

    fn_ids = [x[0] for x in fn_facs]
    n = len(fn_ids)
    if (n == 0 or
        all((is_lda(x) for x in fn_ids))):
        exc_jvp = (vxc[0] - exc) / rho * rho_t
        vxc_jvp = (fxc[0] * rho_t, None, None, None)
        #vxc_jvp = ((fxc[0] - 2. * (vxc[0] - exc) / rho) * rho_t, None, None, None)
    elif any((is_meta_gga(x) for x in fn_ids)):
        raise NotImplementedError
    else:
        exc_jvp = (vxc[0] - exc) / rho[0] * rho_t[0]
        exc_jvp += vxc[1] / rho[0] * 2. * jnp.einsum('np,np->p', rho[1:4], rho_t[1:4])

        vrho1 = fxc[0] * rho_t[0] + fxc[1] * 2. * jnp.einsum('np,np->p', rho[1:4], rho_t[1:4])
        vsigma1 = fxc[1] * rho_t[0] + fxc[2] * 2. * jnp.einsum('np,np->p', rho[1:4], rho_t[1:4])
        vxc_jvp = (vrho1, vsigma1, None, None)

    if deriv == 1:
        fxc = kxc = fxc_jvp = kxc_jvp = None
    return (exc, vxc, fxc, kxc), (exc_jvp, vxc_jvp, fxc_jvp, kxc_jvp)
