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
  LineCurrentWholeSpace
  MagneticDipoleWholeSpace
  MagneticPoleWholeSpace
  CircularLoopWholeSpace
  ElectrostaticSphere
  MagnetostaticSphere
  PointCurrentWholeSpace
  PointCurrentHalfSpace
  DipoleHalfSpace
  MagneticPrism
"""

from geoana.em.static.sphere import (
    ElectrostaticSphere,
    MagnetostaticSphere,
)

from geoana.em.static.wholespace import (
    MagneticDipoleWholeSpace,
    MagneticPoleWholeSpace,
    CircularLoopWholeSpace,
    PointCurrentWholeSpace,
    LineCurrentWholeSpace
)

from geoana.em.static.halfspace import (
    PointCurrentHalfSpace,
    DipoleHalfSpace
)

from geoana.em.static.freespace import MagneticPrism


LineCurrentFreeSpace = LineCurrentWholeSpace

