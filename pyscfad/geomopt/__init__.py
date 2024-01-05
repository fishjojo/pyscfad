
def optimize(*args, **kwargs):
    try:
        from . import geometric_solver as geom
    except ImportError as err:
        raise ImportError('Unable to import geometric.') from err
    return geom.kernel(*args, **kwargs)
