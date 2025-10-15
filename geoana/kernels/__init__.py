"""
===========================================
Kernel Functions (:mod:`geoana.kernels`)
===========================================
.. currentmodule:: geoana.kernels

This module contains kernals for (semi-)analytic geophysical responses
These are optionally available as compiled extensions with fallback
`numpy` implementations.

Inverse Distance Integral Kernels
==================================
Kernels used to evaluate analytic integrals of the
1/r volume integrand (and it's derivatives)

Note that if `numba` is installed and `geoana` has compiled
extensions, these are callable in no-python mode.

.. autosummary::
  :toctree: generated/

  prism_f
  prism_fz
  prism_fzz
  prism_fzx
  prism_fzy
  prism_fzzz
  prism_fxxy
  prism_fxxz
  prism_fxyz

Layered Electromagnetic Reflection Kernels
==========================================
.. autosummary::
  :toctree: generated/

  rTE_forward
  rTE_gradient

"""
from geoana.kernels.tranverse_electric_reflections import rTE_forward, rTE_gradient
from geoana.kernels.potential_field_prism import (
    prism_f,
    prism_fz,
    prism_fzz,
    prism_fzx,
    prism_fzy,
    prism_fzzz,
    prism_fxxy,
    prism_fxxz,
    prism_fxyz,
)
