import numpy as np
from pyscfad.lib import numpy as jnp

def fft(f, mesh):
    '''
    3D FFT with jax.numpy backend
    '''
    if f.size == 0:
        return np.zeros_like(f)

    f3d = f.reshape(-1, *mesh)
    assert(f3d.shape[0] == 1 or f[0].size == f3d[0].size)
    g3d = jnp.fft.fftn(f3d, axes=(1,2,3))
    ngrids = np.prod(mesh)
    if f.ndim == 1 or (f.ndim == 3 and f.size == ngrids):
        return g3d.ravel()
    else:
        return g3d.reshape(-1, ngrids)

def ifft(g, mesh):
    '''
    3D inverse FFT with jax.numpy backend
    '''
    if g.size == 0:
        return np.zeros_like(g)

    g3d = g.reshape(-1, *mesh)
    assert(g3d.shape[0] == 1 or g[0].size == g3d[0].size)
    f3d = jnp.fft.ifftn(g3d, axes=(1,2,3))
    ngrids = np.prod(mesh)
    if g.ndim == 1 or (g.ndim == 3 and g.size == ngrids):
        return f3d.ravel()
    else:
        return f3d.reshape(-1, ngrids)

def fftk(f, mesh, expmikr):
    r'''Perform the 3D FFT of a real-space function which is (periodic*e^{ikr}).

    fk(k+G) = \sum_r fk(r) e^{-i(k+G)r} = \sum_r [f(k)e^{-ikr}] e^{-iGr}
    '''
    return fft(f*expmikr, mesh)


def ifftk(g, mesh, expikr):
    r'''Perform the 3D inverse FFT of f(k+G) into a function which is (periodic*e^{ikr}).

    fk(r) = (1/Ng) \sum_G fk(k+G) e^{i(k+G)r} = (1/Ng) \sum_G [fk(k+G)e^{iGr}] e^{ikr}
    '''
    return ifft(g, mesh) * expikr
