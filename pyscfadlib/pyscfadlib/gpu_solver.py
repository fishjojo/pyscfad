try:
    from jax import ffi
except ImportError:
    from jax.extend import ffi

from pyscfadlib._cuda_plugin import import_plugin_module

# Load the solver module from the CUDA plugin matching jax's CUDA version
# (pyscfad-cuda12-plugin / pyscfad-cuda13-plugin / ...).
_cusolver = import_plugin_module("_solver")

if _cusolver:
    for _name, _value in _cusolver.registrations().items():
        ffi.register_ffi_target(
            _name,
            _value,
            platform="CUDA",
            api_version=(1 if _name.endswith("_ffi") else 0),
        )

