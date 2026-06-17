"""Locate and import the installed pyscfad CUDA plugin.

There is one plugin per CUDA major version (``pyscfad-cuda12-plugin``,
``pyscfad-cuda13-plugin``, ...). The correct one is the plugin whose CUDA major
matches the CUDA runtime jax is using, because the plugin's compiled modules link
that major's CUDA libraries (e.g. ``libcusolver.so.11`` for cu12 vs
``libcusolver.so.12`` for cu13). We detect jax's CUDA major and prefer the
matching plugin, falling back to whichever plugin is importable.
"""
import importlib

# Known plugin packages keyed by CUDA major version.
_PLUGIN_PACKAGES = {
    12: "pyscfad_cuda12_plugin",
    13: "pyscfad_cuda13_plugin",
}


def jax_cuda_major():
    """CUDA major version of the running jax, or ``None`` if undeterminable."""
    # Primary: the CUDA runtime jaxlib actually loaded (most accurate).
    try:
        from jax._src.lib import cuda_versions
        if cuda_versions is not None:
            return cuda_versions.cuda_runtime_get_version() // 1000
    except Exception:
        pass
    # Fallback: the installed jax CUDA plugin (public packaging metadata).
    try:
        from importlib import metadata
        for dist in metadata.distributions():
            name = (dist.metadata["Name"] or "")
            if name.startswith("jax-cuda") and name.endswith("-plugin"):
                major = name[len("jax-cuda"):-len("-plugin")]
                if major.isdigit():
                    return int(major)
    except Exception:
        pass
    return None


def _packages_in_preference_order():
    pkgs = list(_PLUGIN_PACKAGES.values())
    major = jax_cuda_major()
    preferred = _PLUGIN_PACKAGES.get(major)
    if preferred is not None:
        pkgs = [preferred] + [p for p in pkgs if p != preferred]
    return pkgs


def import_plugin_module(submodule):
    """Import a submodule (e.g. ``"_solver"``/``"_cuint"``) from the matching
    CUDA plugin. Returns the module, or ``None`` if no plugin is installed.
    """
    for pkg in _packages_in_preference_order():
        try:
            return importlib.import_module(f".{submodule}", package=pkg)
        except ImportError:
            continue
    return None
