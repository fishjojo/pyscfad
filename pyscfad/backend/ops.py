from .config import get_backend

__all__ = [
    'is_array',
    'isarray',
    'is_tensor',
    'to_numpy',
    'stop_gradient',
    'stop_grad',
    'stop_trace',
    'class_as_pytree_node',
    'custom_jvp',
    'jit',
    'vmap',
    'index',
    'index_update',
    'index_add',
    'index_mul',
]

def __getattr__(name):
    return getattr(get_backend(), name)

def is_array(x):
    return get_backend().is_array(x)

is_tensor = isarray = is_array

def to_numpy(x):
    return get_backend().to_numpy(x)

def stop_gradient(x):
    return get_backend().stop_gradient(x)

stop_grad = stop_gradient

def stop_trace(fn):
    """Convenient wrapper to call functions with arguments
    detached from the graph.
    """
    def wrapped_fn(*args, **kwargs):
        args_no_grad = [stop_grad(arg) for arg in args]
        kwargs_no_grad = {k : stop_grad(v) for k, v in kwargs.items()}
        return fn(*args_no_grad, **kwargs_no_grad)
    return wrapped_fn

def class_as_pytree_node(cls, leaf_names, num_args=0):
    return get_backend().class_as_pytree_node(cls, leaf_names, num_args=num_args)

def jit(obj, **kwargs):
    return get_backend().jit(obj, **kwargs)

def vmap(fun, in_axes=0, out_axes=0, chunk_size=None, signature=None):
    return get_backend().vmap(fun, in_axes=in_axes, out_axes=out_axes,
                              chunk_size=chunk_size, signature=signature)

def index_update(x, idx, y):
    return get_backend().index_update(x, idx, y)

def index_add(x, idx, y):
    return get_backend().index_add(x, idx, y)

def index_mul(x, idx, y):
    return get_backend().index_mul(x, idx, y)

