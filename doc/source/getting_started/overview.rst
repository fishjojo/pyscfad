.. _overview:

================
Package overview
================

pyscfad is an extension package to pyscf, which is meant to serve as
a framework for differentiable quantum chemistry calculations.
As of version 0.1.4, pyscfad uses jax as its backend to perform
automatic differentiation. However, a pytorch backend will be
added in future releases.

Purpose
-------

pyscfad allows the commonly used electron integrals to be differentiable
(up to certain orders). This makes implementing differentiable
quantum chemistry methods straightforward.
In most scenarios, one can simply replace ``numpy`` with ``jax.numpy``,
if the method can be implemented mainly with Python (see many examples in pyscf).
pyscfad also provides a few prebuilt quantum chemistry methods,
including Hartree-Fock, MP2, coupled cluster methods, *etc.*,
as illustrations and for research usages.
However, the main purpose of pyscfad is to provide a framework within which
users can develop their own methods.

Development
-----------

Depending on the usage, many new features may be needed for the pyscfad core.
If you find an interesting problem where autodiff can be useful,
please don't hesitate to submit an issue on the Github page.

License
-------

.. literalinclude:: ../../../LICENSE
