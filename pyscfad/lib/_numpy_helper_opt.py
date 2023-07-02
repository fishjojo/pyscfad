from functools import partial
import ctypes
import numpy
from jax import custom_vjp
from pyscf import lib as pyscf_lib

_np_helper = pyscf_lib.load_library('libnp_helper')

@partial(custom_vjp, nondiff_argnums=(1,2,3))
def _unpack_tril(tril, filltriu=1, axis=-1, out=None):
    return pyscf_lib.unpack_tril(tril, filltriu=filltriu, axis=axis, out=out)

def _unpack_tril_fwd(tril, filltriu, axis, out):
    out = _unpack_tril(tril, filltriu, axis, out)
    return out, ()

def _unpack_tril_bwd(filltriu, axis, out,
                     res, ybar):
    ybar = numpy.asarray(ybar, order='C')
    if ybar.ndim == 2:
        count, nd = 1, ybar.shape[-1]
        n2 = nd*(nd+1)//2
        shape = (n2,)
    elif ybar.ndim == 3:
        if axis == 0 or axis == -2:
            raise KeyError
        else:
            count = ybar.shape[0]
            nd = ybar.shape[-1]
        n2 = nd*(nd+1)//2
        shape = (count, n2)
    else:
        raise KeyError

    vjp = numpy.empty(shape, ybar.dtype)
    if ybar.dtype == numpy.double:
        fn = _np_helper.NPdunpack_tril_2d_vjp
    else:
        raise NotImplementedError
    fn(ctypes.c_int(count), ctypes.c_int(nd),
       vjp.ctypes.data_as(ctypes.c_void_p),
       ybar.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(filltriu))
    return (vjp,)

_unpack_tril.defvjp(_unpack_tril_fwd, _unpack_tril_bwd)
