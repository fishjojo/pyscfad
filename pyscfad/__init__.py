"""
PySCF with auto-differentiation
"""
from pyscfad.version import __version__

#from pyscfad import util
#from pyscfad import implicit_diff

from pyscfad._src._config import (
    config,
    config_update
)

#from jax import config as _jconf
#_jconf.update("jax_enable_x64", True)
