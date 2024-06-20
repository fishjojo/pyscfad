"""
Default configurations related to the backend.
"""
import os
import sys
import json
import importlib
import contextlib
import threading

# default
_FLOATX = 'float64'
_BACKEND = 'jax'

_allowed_floatx = ('float32', 'float64')
_allowed_backend = ('numpy', 'cupy', 'jax', 'torch')
_floatx = _backend = None

if 'PYSCFAD_HOME' in os.environ:
    _base_dir = os.environ['PYSCFAD_HOME']
    _PYSCFAD_DIR = os.path.expanduser(_base_dir)
else:
    _base_dir = os.path.expanduser('~')
    if not os.access(_base_dir, os.W_OK):
        _base_dir = '/tmp'
    _PYSCFAD_DIR = os.path.join(_base_dir, '.pyscfad')

_config_path = os.path.expanduser(os.path.join(_PYSCFAD_DIR, "pyscfad.json"))
if os.path.exists(_config_path):
    try:
        with open(_config_path) as f:
            _config = json.load(f)
    except ValueError:
        _config = {}

    _floatx = _config.get('floatx', None)
    _backend = _config.get('backend', None)

# NOTE environment variables overwrite configure file
if 'PYSCFAD_FLOATX' in os.environ:
    _floatx = os.environ['PYSCFAD_FLOATX']
if 'PYSCFAD_BACKEND' in os.environ:
    _backend = os.environ['PYSCFAD_BACKEND']

if _floatx in _allowed_floatx:
    _FLOATX = _floatx
if _backend in _allowed_backend:
    _BACKEND = _backend

if not os.path.exists(_PYSCFAD_DIR):
    try:
        os.makedirs(_PYSCFAD_DIR)
    except OSError:
        pass

if not os.path.exists(_config_path):
    _config = {
        "floatx": _FLOATX,
        "backend": _BACKEND,
    }
    try:
        with open(_config_path, "w") as f:
            f.write(json.dumps(_config, indent=4))
    except IOError:
        pass

del (_floatx, _backend)
del (os, sys, json)

def default_backend():
    return _BACKEND

def default_floatx():
    return _FLOATX


#---------------- dynamic backend update ----------------#

_current_backend = None
_backend_cache = {}

def set_backend(backend_name):
    if not backend_name in _allowed_backend:
        raise KeyError(f"Required backend {backend_name} is not supported.")

    with threading.RLock():
        global _current_backend
        if backend_name in _backend_cache:
            _current_backend = _backend_cache[backend_name]
        else:
            try:
                module = importlib.import_module(f"pyscfad.backend._{backend_name}").backend
            except Exception:
                raise RuntimeError("Failed setting backend {backend_name}.")
            _backend_cache[backend_name] = module
            _current_backend = module

def get_backend():
    return _current_backend

@contextlib.contextmanager
def with_backend(backend_name):
    with threading.RLock():
        global _current_backend
        previous_backend = _current_backend
        set_backend(backend_name)
        try:
            yield
        finally:
            _current_backend = previous_backend

