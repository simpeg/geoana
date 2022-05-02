"""
============================================================
Frequency-Domain Electromagnetics (:mod:`geoana.em.fdem`)
============================================================
.. currentmodule:: geoana.em.fdem

The ``geoana.em.fdem`` module contains simulation classes for solving
basic frequency-domain electromagnetic problems.

Simulation Classes
==================
.. autosummary::
  :toctree: generated/

  BaseFDEM
  ElectricDipoleWholeSpace
  MagneticDipoleWholeSpace
  MagneticDipoleHalfSpace
  MagneticDipoleLayeredHalfSpace

Utility Functions
=================

.. autosummary::
  :toctree: generated/

  omega
  wavenumber
  skin_depth
  sigma_hat
  vertical_magnetic_field_horizontal_loop
  vertical_magnetic_flux_horizontal_loop
"""
from geoana.em.fdem.base import (
    omega, wavenumber, skin_depth, sigma_hat, BaseFDEM
)

from geoana.em.fdem.wholespace import (
    ElectricDipoleWholeSpace, MagneticDipoleWholeSpace
)

from geoana.em.fdem.halfspace import MagneticDipoleHalfSpace

from geoana.em.fdem.layered import MagneticDipoleLayeredHalfSpace

from geoana.em.fdem.simple_functions import (
    vertical_magnetic_field_horizontal_loop, vertical_magnetic_flux_horizontal_loop
)
