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
  MagnetostaticSphere
  PointCurrentWholeSpace
  PointCurrentHalfSpace
  DipoleHalfSpace
"""

from geoana.em.static.sphere import ElectrostaticSphere

from geoana.em.static.sphere import MagnetostaticSphere

from geoana.em.static.wholespace import (
    MagneticDipoleWholeSpace,
    MagneticPoleWholeSpace,
    CircularLoopWholeSpace,
    PointCurrentWholeSpace
)

from geoana.em.static.halfspace import (
    PointCurrentHalfSpace,
    DipoleHalfSpace
)

from geoana.em.static.freespace import LineCurrentFreeSpace, MagneticPrism
