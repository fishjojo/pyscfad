def stop_gradient(x):
    return x

class custom_jvp:
    """Fake ``custom_jvp`` that does nothing.
    """
    def __init__(self, fun, *args, **kwargs):
        self.fun = fun
        self.jvp = None

    def defjvp(self, jvp):
        self.jvp = jvp
        return jvp

    def __call__(self, *args, **kwargs):
        return self.fun(*args, **kwargs)

def jit(fun, **kwargs):
    return fun

def while_loop(cond_fun, body_fun, init_val):
    val = init_val
    while cond_fun(val):
        val = body_fun(val)
    return val

# TODO deprecate these
class _Indexable(object):
    # pylint: disable=line-too-long
    """
    see https://github.com/google/jax/blob/97d00584f8b87dfe5c95e67892b54db993f34486/jax/_src/ops/scatter.py#L87
    """
    __slots__ = ()

    def __getitem__(self, idx):
        return idx

index = _Indexable()

def index_update(x, idx, y):
    x[idx] = y
    return x

def index_add(x, idx, y):
    x[idx] += y
    return x

def index_mul(x, idx, y):
    x[idx] *= y
    return x

