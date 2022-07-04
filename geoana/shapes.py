"""
======================================================
Shapes (:mod:`geoana.shapes`)
======================================================
.. currentmodule:: geoana.shapes

The ``geoana.shapes`` module contains the basic geometric building blocks for
three dimensional objects.

Shape Classes
=============
.. autosummary::
  :toctree: generated/

  BasePrism
"""
import numpy as np


class BasePrism:
    """Class for basic geometry of a prism.

    The ``BasePrism`` class is used to define and validate basic geometry of an axis
    aligned prism in three dimensions.

    Parameters
    ----------
    min_location : (3,) numpy.ndarray of float
        Minimum location triple of the axis aligned prism
    max_location : (3,) numpy.ndarray of float
        Maximum location triple of the axis aligned prism
    """

    def __init__(self, min_location, max_location, **kwargs):
        super().__init__(**kwargs)
        self.min_location = min_location
        self.max_location = max_location
        if np.any(self.max_location <= self.min_location):
            raise ValueError("Max location must be strictly greater than the minimum location")

    @property
    def min_location(self):
        """Location of the point mass.

        Returns
        -------
        (3) numpy.ndarray of float
            Location of the point mass in meters.
        """
        return self._min_location

    @min_location.setter
    def min_location(self, vec):

        try:
            vec = np.asarray(vec, dtype=float)
        except:
            raise TypeError(f"location must be array_like of float, got {type(vec)}")

        vec = np.squeeze(vec)
        if vec.shape != (3,):
            raise ValueError(
                f"location must be array_like with shape (3,), got {vec.shape}"
            )

        self._min_location = vec

    @property
    def max_location(self):
        """Location of the point mass.

        Returns
        -------
        (3) numpy.ndarray of float
            Location of the point mass in meters.
        """
        return self._max_location

    @max_location.setter
    def max_location(self, vec):

        try:
            vec = np.asarray(vec, dtype=float)
        except:
            raise TypeError(f"location must be array_like of float, got {type(vec)}")

        vec = np.squeeze(vec)
        if vec.shape != (3,):
            raise ValueError(
                f"location must be array_like with shape (3,), got {vec.shape}"
            )

        self._max_location = vec

    @property
    def volume(self):
        """ The volume of the prism

        Returns
        -------
        float
        """
        return np.prod(self.max_location - self.min_location)

    @property
    def location(self):
        """ The center of the prism

        Returns
        -------
        (3,) numpy.ndarray of float
        """
        return 0.5 * (self.min_location + self.max_location)

    def _eval_def_int(self, func, x, y, z, cycle=0):
        "evaluate a definite integral (func) over the prism at x, y, z locations"

        x_min, y_min, z_min = self.min_location
        x_max, y_max, z_max = self.max_location

        x_min = x_min - x
        y_min = y_min - y
        z_min = z_min - z

        x_max = x_max - x
        y_max = y_max - y
        z_max = z_max - z

        for i in range(cycle):
            x_min, y_min, z_min = y_min, z_min, x_min
            x_max, y_max, z_max = y_max, z_max, x_max

        v000 = func(x_min, y_min, z_min)
        v001 = func(x_min, y_min, z_max)
        v010 = func(x_min, y_max, z_min)
        v011 = func(x_min, y_max, z_max)
        v100 = func(x_max, y_min, z_min)
        v101 = func(x_max, y_min, z_max)
        v110 = func(x_max, y_max, z_min)
        v111 = func(x_max, y_max, z_max)

        val = (v111 - v110 - v101 + v100 - v011 + v010 + v001 - v000)
        return val
