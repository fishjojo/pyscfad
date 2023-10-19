from pyscfad import backend

__all__ = [
    'stop_gradient',
    'stop_grad',
    'convert_to_numpy',
    'convert_to_tensor',
]

def stop_gradient(x):
    return backend.core.stop_gradient(x)

stop_grad = stop_gradient

def convert_to_numpy(x):
    return backend.core.convert_to_numpy(x)

def convert_to_tensor(x, dtype=None, **kwargs):
    return backend.core.convert_to_tensor(x, dtype=dtype, **kwargs)
