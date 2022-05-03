"""
======================================================
Time-Domain Electromagnetics (:mod:`geoana.em.tdem`)
======================================================
.. currentmodule:: geoana.em.tdem

The ``geoana.em.tdem`` module contains simulation classes for solving
basic time-domain electromagnetic problems.

Simulation Classes
==================
.. autosummary::
  :toctree: generated/

  BaseTDEM
  ElectricDipoleWholeSpace
  VerticalMagneticDipoleHalfSpace

Utility Functions
=================

.. autosummary::
  :toctree: generated/

  peak_time
  diffusion_distance
  theta
  vertical_magnetic_field_horizontal_loop
  vertical_magnetic_flux_horizontal_loop
  vertical_magnetic_field_time_deriv_horizontal_loop
  vertical_magnetic_flux_time_deriv_horizontal_loop
  magnetic_field_vertical_magnetic_dipole
  magnetic_field_time_deriv_magnetic_dipole
  magnetic_flux_vertical_magnetic_dipole
  magnetic_flux_time_deriv_magnetic_dipole
  
"""
from geoana.em.tdem.base import peak_time, diffusion_distance, theta, BaseTDEM

from geoana.em.tdem.wholespace import ElectricDipoleWholeSpace

from geoana.em.tdem.halfspace import VerticalMagneticDipoleHalfSpace

from geoana.em.tdem.simple_functions import (
    vertical_magnetic_field_horizontal_loop,
    vertical_magnetic_flux_horizontal_loop,
    vertical_magnetic_field_time_deriv_horizontal_loop,
    vertical_magnetic_flux_time_deriv_horizontal_loop,
    magnetic_field_vertical_magnetic_dipole,
    magnetic_field_time_deriv_magnetic_dipole,
    magnetic_flux_vertical_magnetic_dipole,
    magnetic_flux_time_deriv_magnetic_dipole,
)
