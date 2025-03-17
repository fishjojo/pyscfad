# Copyright 2021 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@xla//xla/tsl:tsl.bzl", _pybind_extension = "tsl_pybind_extension_opensource")

nanobind_extension = _pybind_extension

PLATFORM_TAGS_DICT = {
    ("Linux", "x86_64"): ("manylinux2014", "x86_64"),
    ("Linux", "aarch64"): ("manylinux2014", "aarch64"),
}

def pytype_strict_library(name, pytype_srcs = [], **kwargs):
    data = pytype_srcs + (kwargs["data"] if "data" in kwargs else [])
    new_kwargs = {k: v for k, v in kwargs.items() if k != "data"}
    native.py_library(name = name, data = data, **new_kwargs)
