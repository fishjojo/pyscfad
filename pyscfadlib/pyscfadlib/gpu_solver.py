import importlib

try:
    from jax import ffi
except ImportError:
    from jax.extend import ffi

try:
    _cusolver = importlib.import_module(
        ".pyscfad_cusolver", package="pyscfadlib",
    )
except ImportError:
    _cusolver = None

if _cusolver:
    for _name, _value in _cusolver.registrations().items():
        ffi.register_ffi_target(
            _name,
            _value,
            platform="CUDA",
            api_version=(1 if _name.endswith("_ffi") else 0),
        )
