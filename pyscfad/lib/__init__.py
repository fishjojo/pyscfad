from jax import numpy
from jax.config import config as jax_config

from pyscfad.lib import ops
from pyscfad.lib.ops import *
from pyscfad.lib import jax_helper
from pyscfad.lib.jax_helper import *

jax_config.update("jax_enable_x64", True)
