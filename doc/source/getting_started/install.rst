.. _install:

============
Installation
============


Python version support
----------------------

Officially Python 3.11, 3.12, 3.13, and 3.14.

Installing from PyPI
--------------------

pyscfad can be installed via pip from `PyPI <https://pypi.org/project/pyscfad/>`_.
Choose the command matching your hardware:

====================  =================================
Hardware              Installation
====================  =================================
CPU                   ``pip install pyscfad``
NVIDIA GPU (CUDA 12)  ``pip install "pyscfad[cuda12]"``
NVIDIA GPU (CUDA 13)  ``pip install "pyscfad[cuda13]"``
====================  =================================

The ``cuda12``/``cuda13`` extras pull in ``jax`` with the matching CUDA support and the
``pyscfad-cuda<major>-plugin`` wheel, which interfaces with the NVIDIA cuSOLVER/cuBLAS
libraries. Pick the extra whose CUDA major matches your driver/toolkit.

Supported platforms
--------------------

Prebuilt wheels are published for the following platforms:

============================  ===  ==========
Platform                      CPU  NVIDIA GPU
============================  ===  ==========
Linux, x86_64                 yes  yes
Linux, aarch64                yes  yes
macOS, Apple silicon (arm64)  yes  n/a
macOS, Intel (x86_64)         no   n/a
Windows, x86_64               no   no
Windows WSL2, x86_64          yes  yes
============================  ===  ==========

On platforms without a prebuilt wheel, ``pyscfadlib`` can still be compiled from source
(see `Installing from source`_).

Installing from source
----------------------

The source code of pyscfad can be obtained by cloning the `Github repository <https://github.com/fishjojo/pyscfad>`_.

.. code::

   cd $HOME
   git clone https://github.com/fishjojo/pyscfad.git

The main part of pyscfad is pure Python.
One can simply add the top directory of pyscfad to the environment variable ``PYTHONPATH``.

.. code::

   export PYTHONPATH=$HOME/pyscfad:$PYTHONPATH

Alternatively, one can install pyscfad locally via pip by running
the following command at the top directory of pyscfad.

.. code::

   cd $HOME/pyscfad
   pip install -e .

Installing pyscfadlib (CPU only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pyscfadlib is the C extension to pyscfad that provides efficient gradient backpropagation implementations.
Similarly, one can install pyscfadlib locally via pip.

.. code::

   cd $HOME/pyscfad/pyscfadlib
   pip install -e .

Or one can manually compile the C code, and then add pyscfadlib to ``PYTHONPATH``.

.. code::

   cd $HOME/pyscfad/pyscfadlib/pyscfadlib
   mkdir build
   cd build
   cmake ..
   make
   export PYTHONPATH=$HOME/pyscfad/pyscfadlib:$PYTHONPATH

.. note::

    For Mac with ARM64 architectures, one needs to set the environment variable
    ``CMAKE_OSX_ARCHITECTURES=arm64``.

Compiling the CUDA plugin (NVIDIA GPUs)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``pyscfad-cuda12-plugin`` / ``pyscfad-cuda13-plugin`` packages are only needed when
running on NVIDIA GPUs. They are built with CMake and provide the ``_solver``
(cuSOLVER) and ``_cuint`` modules.

Building from source requires a CUDA toolkit on ``PATH`` whose major matches the wheel
(CUDA 12.8+ for the cuda12 plugin, 13.x for the cuda13 plugin), together with the
``cmake``, ``nanobind``, ``jax``, and ``build`` Python packages. From the ``pyscfadlib``
directory:

.. code::

   cd $HOME/pyscfad/pyscfadlib
   python plugins/cuda/build_plugin.py --cuda-major 13   # or --cuda-major 12
   pip install dist/pyscfad_cuda13_plugin*.whl

``build_plugin.py`` drives ``plugins/cuda/CMakeLists.txt``, which builds the nanobind
modules and fetches the ``cuint`` kernels from GitHub. The CUDA major (default: detected
from ``nvcc``) selects both the wheel name and the matching ``nvidia-*`` runtime
dependencies. Device architectures default to up to ``sm_120`` for the CUDA major;
override them with ``--cuda-arch``, e.g.

.. code::

   python plugins/cuda/build_plugin.py --cuda-major 12 --cuda-arch "70-real;80-real;90-real"

To also install the CUDA runtime libraries (cuSOLVER, cuBLAS, etc.) from PyPI alongside
the plugin, install the wheel with the ``with_cuda`` extra:

.. code::

   pip install "dist/pyscfad_cuda13_plugin*.whl[with_cuda]"

Dependencies
------------

pyscfad requires the following dependencies.

==========================================================  ==================
Package                                                     supported versions
==========================================================  ==================
`jax <https://jax.readthedocs.io/en/latest/>`_              >=0.9.1,<0.11
`pyscfadlib <https://pypi.org/project/pyscfadlib/>`_        >=0.3
`pyscf <https://pyscf.org/>`_                               >=2.3
`pyscf-properties <https://github.com/pyscf/properties>`_   >=0.1
==========================================================  ==================

