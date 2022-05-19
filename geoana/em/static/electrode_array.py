import numpy as np

__all__ = [
    "PointCurrentFourElectrodeArray"
]


class PointCurrentFourElectrodeArray:
    """Class for a point current in a four electrode array.

    The ``PointCurrentFourElectrodeArray`` class is used to analytically compute the
    potentials, current densities and electric fields within a four electrode array due to a point current.

    Parameters
    ----------
    current : float
        Electrical current in the point current (A). Default is 1A.
    rho : float
        Resistivity in the point current (:math:`\\Omega \\cdot m`).
    location : array_like, optional
        Location at which we are observing in 3D space (m). Default is (0, 0, 0).
    """