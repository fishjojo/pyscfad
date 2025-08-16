from pyscfad.backend import get_backend

if get_backend().__name__ == "jax.numpy":
    from jax.scipy.special import *
else:
    from scipy.special import *
