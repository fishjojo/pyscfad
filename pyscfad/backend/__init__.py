"""
The module oversees the backend.
"""
from pyscfad.backend.config import (
    default_backend,
    set_backend,
    get_backend,
    with_backend,
)
from pyscfad.backend import numpy as numpy
from pyscfad.backend import ops as ops

set_backend(default_backend())

__all__ = [
    'set_backend',
    'get_backend',
    'with_backend',
    'numpy',
    'ops',
]

