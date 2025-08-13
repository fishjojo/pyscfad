# Copyright 2021-2025 Xing Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

class _Config:
    def __init__(self):
        self.values = {}

    def update(self, name, val):
        if not self.exist(name):
            raise KeyError(f'Invalid configuration key: {name}.')
        self._setter(name, val)

    def set_default(self, name, val):
        if not name.startswith('pyscfad_'):
            raise KeyError('PySCFAD configuration keys start with \'pyscfad_\'.')
        self.values[name] = val
        self._setter(name, val)

    def _setter(self, name, val):
        setattr(self, name[8:], val)

    def exist(self, name):
        return hasattr(self, name[8:])

    def reset(self):
        for name, val in self.values.items():
            self._setter(name, val)

config = _Config()

config.set_default('pyscfad_scf_implicit_diff', False)
config.set_default('pyscfad_ccsd_implicit_diff', False)
#config.set_default('pyscfad_ccsd_checkpoint', False)
config.set_default('pyscfad_moleintor_opt', False)


class config_update:
    def __init__(self, name, value):
        self.name = name
        self.val_orig = getattr(config, name[8:])
        self.val_new = value

    def __enter__(self):
        config.update(self.name, self.val_new)

    def __exit__(self, *exc):
        config.update(self.name, self.val_orig)
