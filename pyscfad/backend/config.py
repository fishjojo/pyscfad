import os
import sys
import json

# default
_FLOATX = 'float64'
_EPSILON = sys.float_info.epsilon
_BACKEND = 'jax'

_allowed_floatx = ('float16', 'float32', 'float64')
_allowed_backend = ('numpy', 'jax', 'torch', 'tensorflow')

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
    if _floatx in _allowed_floatx:
        _FLOATX = _floatx
    _epsilon = _config.get('epsilon', None)
    if isinstance(_epsilon, float):
        _EPSILON = _epsilon
    _backend = _config.get('backend', None)
    if _backend in _allowed_backend:
        _BACKEND = _backend

if 'PYSCFAD_BACKEND' in os.environ:
    _backend = os.environ['PYSCFAD_BACKEND']
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
        "epsilon": _EPSILON,
        "backend": _BACKEND,
    }
    try:
        with open(_config_path, "w") as f:
            f.write(json.dumps(_config, indent=4))
    except IOError:
        pass

if _BACKEND in ('torch', 'tensorflow'):
    # Keras configuration
    os.environ["KERAS_BACKEND"] = _BACKEND
    try:
        import keras_core as keras
    except ImportError as err:
        raise ImportError('Unable to import keras_core.') from err
    keras.config.set_floatx(_FLOATX)
    keras.config.set_epsilon(_EPSILON)
    del keras

del (os, sys, json)

def backend():
    return _BACKEND

def floatx():
    return _FLOATX
