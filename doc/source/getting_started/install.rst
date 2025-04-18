.. _install:

============
Installation
============


Python version support
----------------------

Officially Python 3.10, 3.11, 3.12, and 3.13.

Installing from PyPI
--------------------

pyscfad can be installed via pip from `PyPI <https://pypi.org/project/pyscfad/>`_.

.. code::

   pip install pyscfad


To install the CUDA compatible version, run

.. code::

    pip install pyscfad[cuda12]

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

Installing pyscfadlib
~~~~~~~~~~~~~~~~~~~~~

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

Dependencies
------------

pyscfad requires the following dependencies.

=====================================================  ==================
Package                                                supported versions
=====================================================  ==================
`numpy <https://numpy.org>`_                           >=1.17
`scipy <https://scipy.org>`_                           >=1.7
`h5py <https://www.h5py.org/>`_                        >=2.7
`jax <https://jax.readthedocs.io/en/latest/>`_         ==0.4.35
`pyscf <https://pyscf.org/>`_                          >=2.3.0
`pyscfadlib <https://pypi.org/project/pyscfadlib/>`_   >=0.1.4
=====================================================  ==================

.. note::

    Since jax version 0.4.36, the tracing machinery has been modified
    to eliminate data-dependent tracing, which conflicts with pyscfad's flexibility.
    A comprehensive update to ensure compatibility with later jax versions will be
    introduced in the future pyscfad 0.2 release.
