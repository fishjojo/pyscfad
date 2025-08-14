# Copyright 2023 The JAX Authors.
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

# Modified from JAX

import importlib
import os
from setuptools import setup
from setuptools.dist import Distribution

__version__ = None
cuda_version = 0  # placeholder
project_name = f"pyscfad-cuda{cuda_version}-plugin"
package_name = f"pyscfad_cuda{cuda_version}_plugin"

def load_version_module(pkg_path):
  spec = importlib.util.spec_from_file_location(
    'version', os.path.join(pkg_path, 'version.py'))
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module

_version_module = load_version_module(package_name)
__version__ = _version_module.__version__

class BinaryDistribution(Distribution):
  """This class makes 'bdist_wheel' include an ABI tag on the wheel."""

  def has_ext_modules(self):
    return True

setup(
    name=project_name,
    version=__version__,
    description="PySCFAD Plugin for NVIDIA GPUs",
    long_description="",
    long_description_content_type="text/markdown",
    author="Xing Zhang",
    author_email="zhangxing.nju@gmail.com",
    packages=[package_name],
    python_requires=">=3.11",
    extras_require={
      'with_cuda': [
          "nvidia-cublas-cu12>=12.1.3.1",
          "nvidia-cuda-nvcc-cu12>=12.6.85",
          "nvidia-cuda-runtime-cu12>=12.1.105",
          "nvidia-cusolver-cu12>=11.4.5.107",
          "nvidia-cusparse-cu12>=12.1.0.106",
          "nvidia-nvjitlink-cu12>=12.1.105",
      ],
    },
    license="Apache-2.0",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    package_data={
        package_name: [
            "*",
        ],
    },
    zip_safe=False,
    distclass=BinaryDistribution,
)
