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

def cuda_runtime_requirements(cuda_version):
  """PyPI requirements for the CUDA runtime libraries the modules link against.

  NVIDIA changed the package naming at CUDA 13: through CUDA 12 the real
  libraries are suffixed (``nvidia-cublas-cu12``); from CUDA 13 on the suffixed
  packages are empty deprecation stubs and the real libraries live under the
  unsuffixed names, with the CUDA major encoded in the package version
  (``nvidia-cublas`` 13.x). For the unsuffixed names we therefore pin the major
  so a cuda13 wheel cannot resolve a future cuda14 library.
  """
  libs = ["nvidia-cublas", "nvidia-cuda-runtime", "nvidia-cusolver",
          "nvidia-cusparse", "nvidia-nvjitlink"]
  if cuda_version >= 13:
    return [f"{name}>={cuda_version},<{cuda_version + 1}" for name in libs]
  return [f"{name}-cu{cuda_version}" for name in libs]

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
      # CUDA runtime libraries, matched to the wheel's CUDA major version.
      # The compiled modules link these (libcusolver.so.11 for cu12,
      # libcusolver.so.12 for cu13, etc.); CI ships slim wheels that resolve
      # them from these packages at runtime.
      'with_cuda': cuda_runtime_requirements(cuda_version),
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
        "Programming Language :: Python :: 3.14",
    ],
    package_data={
        package_name: [
            "*",
        ],
    },
    zip_safe=False,
    distclass=BinaryDistribution,
)
