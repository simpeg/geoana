"""
======================================================
Static Electromagnetics (:mod:`geoana.em.static`)
======================================================
.. currentmodule:: geoana.em.static

The ``geoana.em.static`` module contains simulation classes for solving
basic electrostatic and magnetostatic problems.


Simulation Classes
==================
.. autosummary::
  :toctree: generated/

  LineCurrentFreeSpace
  MagneticDipoleWholeSpace
  MagneticPoleWholeSpace
  CircularLoopWholeSpace
  ElectrostaticSphere
"""

from geoana.em.static.sphere import ElectrostaticSphere

from geoana.em.static.wholespace import (
    MagneticDipoleWholeSpace,
    MagneticPoleWholeSpace,
    CircularLoopWholeSpace
)

from geoana.em.static.freespace import LineCurrentFreeSpace


