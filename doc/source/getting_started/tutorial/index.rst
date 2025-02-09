.. _tutorial:

=========
Tutorials
=========

When using the jax backend to compute derivatives, the best practice
is to construct a function, where the inputs are variables and the outputs
are objectives (see e.g.,
the `jax document <https://jax.readthedocs.io/en/latest/notebooks/quickstart.html#taking-derivatives-with-grad>`_).
In the following tutorials, we assume using the jax backend.

We present a few examples of using pyscfad. Most of them are included in paper `[1]`_.

.. toctree::
   :maxdepth: 1

   01_scf
   02_oorpa
   03_raman_ccsd
   04_nac_cis

_`[1]` `Differentiable quantum chemistry with PySCF for molecules and materials at the mean-field level and beyond <https://doi.org/10.1063/5.0118200>`_, X. Zhang, G. K.-L. Chan, *J. Chem. Phys.*, **157**, 204801 (2022)
