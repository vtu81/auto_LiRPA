.. auto_LiRPA documentation master file, created by
   sphinx-quickstart on Wed Jul 14 21:56:10 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to auto_LiRPA's documentation!
======================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. autoclass:: auto_LiRPA.BoundedModule

   .. autofunction:: auto_LiRPA.BoundedModule.compute_bounds

.. autoclass:: auto_LiRPA.bound_ops.Bound

   .. autofunction:: auto_LiRPA.bound_ops.Bound.forward
   .. autofunction:: auto_LiRPA.bound_ops.Bound.interval_propagate
   .. autofunction:: auto_LiRPA.bound_ops.Bound.bound_forward
   .. autofunction:: auto_LiRPA.bound_ops.Bound.bound_backward

.. autoclass:: auto_LiRPA.perturbations.Perturbation

   .. autofunction:: auto_LiRPA.perturbations.Perturbation.concretize
   .. autofunction:: auto_LiRPA.perturbations.Perturbation.init

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`

..
   * :ref:`modindex`