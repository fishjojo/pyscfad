class _Config:
    def __init__(self):
        self.values = {}

    def update(self, name, val):
        if not self.exist(name):
            raise KeyError(f'Invalid configuration key: {name}.')
        self._setter(name, val)

    def set_default(self, name, val):
        self._setter(name, val)

    def _setter(self, name, val):
        if not name.startswith('pyscfad_'):
            raise KeyError('PySCFAD configuration keys start with \'pyscfad_\'.')
        self.values[name] = val
        setattr(self, name[8:], val)

    def exist(self, name):
        return hasattr(self, name[8:])

config = _Config()

config.set_default('pyscfad_numpy_backend', 'jax')
config.set_default('pyscfad_scf_implicit_diff', False)
config.set_default('pyscfad_ccsd_implicit_diff', False)
config.set_default('pyscfad_moleintor_opt', False)
