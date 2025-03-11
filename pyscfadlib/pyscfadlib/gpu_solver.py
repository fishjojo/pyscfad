try:
    from jax import ffi
except ImportError:
    from jax.extend import ffi

from pyscfadlib import pyscfad_cusolver as _cusolver

for _name, _value in _cusolver.registrations().items():
    ffi.register_ffi_target(
        _name,
        _value,
        platform="CUDA",
        api_version=(1 if _name.endswith("_ffi") else 0),
    )
