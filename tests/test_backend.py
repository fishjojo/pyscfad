# Copyright 2026 The PySCFAD Authors
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

import pytest
from pyscfad.backend.config import set_backend, _allowed_backend


def test_set_backend_unsupported_lists_supported_backends():
    # An unsupported backend must raise ValueError (conventional for an invalid
    # argument value) with a message that enumerates the supported backends.
    with pytest.raises(ValueError, match="not_a_backend"):
        set_backend("not_a_backend")

    with pytest.raises(ValueError) as excinfo:
        set_backend("not_a_backend")
    message = str(excinfo.value)
    for name in _allowed_backend:
        assert name in message, (
            f"supported backend {name!r} missing from error message: {message!r}"
        )
