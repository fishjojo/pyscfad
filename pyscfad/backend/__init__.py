from pyscfad.backend.config import backend
from pyscfad.backend.config import floatx

if backend() == 'numpy':
    from pyscfad.backend._numpy import *
elif backend() == 'jax':
    try:
        import jax
    except ImportError as err:
        raise ImportError('Unable to import jax.') from err
    from jax import config as jconf
    if floatx() == 'float64':
        jconf.update('jax_enable_x64', True)
    from pyscfad.backend._jax import *
elif backend() == 'torch':
    try:
        import torch
    except ImportError as err:
        raise ImportError('Unable to import torch.') from err
    from pyscfad.backend._torch import *
elif backend() == 'tensorflow':
    raise NotImplementedError('tensorflow API is under development')
