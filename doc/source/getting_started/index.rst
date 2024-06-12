.. _getting_started:

===============
Getting started
===============

Installation
------------

The easiest way to install pyscfad is to
install it via pip from `PyPI <https://pypi.org/project/pyscfad/>`_.

.. code::

   pip install pyscfad

More sophisticated instructions for installation can be found in the :ref:`install` page.


Examples
--------

The examples of using pyscfad can be found in
the `source <https://github.com/fishjojo/pyscfad/tree/main/examples>`_.
In addition, refer to the :ref:`tutorial` page for more tutorials.

.. note::
   For pyscfad with versions <= 0.1.4, one needs to configure pyscf
   in order to perform autodiff calculations.
   In the pyscf configure file (``$HOME/.pyscf_conf.py``), add

   .. code:: python

      pyscfad = True
   
   This will be no longer needed in later versions of pyscfad.

.. toctree::
   :maxdepth: 2
   :hidden:

   install
   overview
   tutorial/index
