"""
PySCF with auto-differentiation
"""
import sys
from pyscfad.version import __version__

from pyscfad._src._config import (
    config,
    config_update
)

# export backend.numpy to pyscfad namespace
# pylint: disable=useless-import-alias
from pyscfad.backend import numpy as numpy
from pyscfad.backend import ops as ops
sys.modules['pyscfad.numpy'] = numpy
sys.modules['pyscfad.ops'] = ops

del sys
