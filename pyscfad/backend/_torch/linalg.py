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

import torch

def cholesky(a, **kwargs):
    a = torch.as_tensor(a)
    return torch.linalg.cholesky(a, **kwargs)

def eigh(a, UPLO='L', **kwargs):
    a = torch.as_tensor(a)
    return torch.linalg.eigh(a, UPLO, **kwargs)

def inv(a, **kwargs):
    a = torch.as_tensor(a)
    return torch.linalg.inv(a, **kwargs)

def norm(x, ord=None, axis=None, keepdims=False, **kwargs):
    x = torch.as_tensor(x)
    return torch.linalg.norm(x, ord, axis, keepdims, **kwargs)
