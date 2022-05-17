"""
=================================================
Base Electromagnetics (:mod:`geoana.em`)
=================================================
.. currentmodule:: geoana.em

The ``geoana.em.base`` module contains base classes whose properties are relevant
to a large range of electromagnetic problems. These classes are not used directly
to compute solutions to electromagnetic problems.


Base Classes
============
.. autosummary::
  :toctree: generated/

  BaseEM
  BaseDipole
  BaseElectricDipole
  BaseMagneticDipole
  BaseLineCurrent

"""
from . import static
from . import fdem
from . import tdem
from .base import BaseEM, BaseDipole, BaseElectricDipole, BaseMagneticDipole, BaseLineCurrent


