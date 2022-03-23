"""
=================================================
Base Electromagnetics (:mod:`geoana.em`)
=================================================
.. currentmodule:: geoana.em

Things about electromagnetics

Base Classes
============
.. autosummary::
  :toctree: generated/

  BaseEM
  BaseDipole
  BaseElectricDipole
  BaseMagneticDipole

"""
from . import static
from . import fdem
from . import tdem
from .base import BaseEM, BaseDipole, BaseElectricDipole, BaseMagneticDipole


