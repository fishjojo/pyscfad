import importlib

try:
    from jax import ffi
except ImportError:
    from jax.extend import ffi

try:
    _cusolver = importlib.import_module(
        "._solver", package="pyscfad_cuda12_plugin",
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
