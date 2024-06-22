import numpy as np

def is_array(x):
    # FIXME should np.generic be included?
    return isinstance(x, (np.ndarray, np.generic))

def to_numpy(x):
    return np.asarray(x)

def convert_to_tensor(x, dtype=None, **kwargs):
    return np.asarray(x, dtype=dtype, **kwargs)

def vmap(fun, in_axes=0, out_axes=0, chunk_size=None, signature=None):
    if not isinstance(out_axes, int):
        raise NotImplementedError

    def vmap_f(*args):
        if isinstance(in_axes, int):
            in_axes_loc = (in_axes,) * len(args)
        else:
            in_axes_loc = in_axes

        if isinstance(in_axes_loc, (list, tuple)):
            excluded = []
            vmap_args = []
            assert len(in_axes_loc) == len(args)
            for i, axis in enumerate(in_axes_loc):
                if axis is None:
                    excluded.append(i)
                    vmap_args.append(args[i])
                elif isinstance(axis, int):
                    vmap_args.append(np.moveaxis(args[i], axis, 0))
                else:
                    raise KeyError
            if len(excluded) > 0:
                excluded = set(excluded)
            else:
                excluded = None

            vfun = np.vectorize(fun, excluded=excluded, signature=signature)
            out = vfun(*vmap_args)
        else:
            raise KeyError

        if out_axes != 0:
            out = np.moveaxis(out, 0, out_axes)
        return out

    return vmap_f

