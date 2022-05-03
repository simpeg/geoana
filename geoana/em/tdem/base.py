from geoana.em.base import BaseEM
import numpy as np
from scipy.constants import mu_0


###############################################################################
#                                                                             #
#                           Utility Functions                                 #
#                                                                             #
###############################################################################


def peak_time(z, sigma, mu=mu_0):
    r"""Compute peak time for a plane wave in a homogeneous electromagnetic media.

    For a particular distance along the propagation path of a planewave in
    a homogeneous media, the `peak time <https://em.geosci.xyz/content/maxwell1_fundamentals/transient_planewaves_homogeneous/peaktime.html>`_
    is the time at which the maximum signal amplitude is observed. In other words,
    it is simple computation for when the peak of the planewave will arrive.

    For the quasistatic case (no electric displacement), the peak time is given by:

    .. math::
        t_{max} = \frac{\mu \sigma z^2}{6}

    where :math:`\mu` is the magnetic permeability, :math:`\sigma` is the electrical
    conductivity and *z* is the propagation distance.

    Notes
    -----
    The inputs values will be broadcasted together following normal numpy rules, and
    will support general shapes. Therefore every input, except for the `secondary` flag,
    can be arrays of the same shape.

    Parameters
    ----------
    z : float or numpy.ndarray
        propagation distance from the planewave source (m)
    sigma : float or numpy.ndarray
        electrical conductivity (S/m)
    mu : float or numpy.ndarray, optional
        magnetic permeability (A/m). Default is the permeability of free-space
        (:math:`\mu_0`)

    Returns
    -------
    float, np.ndarray
        Peak time/times in seconds. The dimensions for return will depend on
        whether floats or arrays were used to define the propagation distance
        and physical properties of the medium.

    """
    return (mu * sigma * z**2)/6.


def diffusion_distance(time, sigma, mu=mu_0):
    r"""Compute diffusion distance for a plane wave in a homogeneous electromagnetic media.

    For a planewave source in a homogeneous media, the `diffusion distance <https://em.geosci.xyz/content/maxwell1_fundamentals/transient_planewaves_homogeneous/peakdistance.html>`_
    is the propagation distance the peak has travelled at a given time. It is sometimes
    referred to as the peak distance. The diffusion distance is a simple computation for
    how far a planewave has travelled after a certain time.

    For the quasistatic case (no electric displacement), the diffusion distance is given by:

    .. math::
        D = \sqrt{\frac{2 t}{\mu \sigma}}

    where :math:`\mu` is the magnetic permeability, :math:`\sigma` is the electrical
    conductivity and *t* is the time.

    Notes
    -----
    The inputs values will be broadcasted together following normal numpy rules, and
    will support general shapes. Therefore every input, except for the `secondary` flag,
    can be arrays of the same shape.

    Parameters
    ----------
    time : float or numpy.ndarray
        propagation time (s)
    sigma : float or numpy.ndarray
        electrical conductivity (S/m)
    mu : float or numpy.ndarray, optional
        magnetic permeability (A/m). Default is the permeability of free-space
        (:math:`\mu_0`)

    Returns
    -------
    float, np.ndarray
        Diffusion distance in meters. The dimensions for return will depend on
        whether floats or arrays were used to define the propagation distance
        and physical properties of the medium.

    """
    return np.sqrt(2*time/(mu*sigma))


def theta(time, sigma, mu=mu_0):
    r"""
    Analog to wavenumber in the frequency domain. See Ward and Hohmann, 1988
    pages 174-175:

    .. math::
        \theta = \sqrt{\frac{\mu \sigma}{4t}}

    Parameters
    ----------
    time : float or numpy.ndarray
        propagation time (s)
    sigma : float or numpy.ndarray
        electrical conductivity (S/m)
    mu : float or numpy.ndarray, optional
        magnetic permeability (A/m). Default is the permeability of free-space
        (:math:`\mu_0`)

    Returns
    -------
    float, np.ndarray
        The theta value for all times.

    """
    return np.sqrt(mu*sigma/(4.*time))


class BaseTDEM(BaseEM):
    r"""
    Base time domain electromagnetic class

    The base TDEM class is contructed to define rudimentary properties and methods
    for time-domain electromagnetics.

    Parameters
    ----------
    time: int, float or (n_time) numpy.ndarray
        Time or times used for all computations in seconds.


    """

    def __init__(self, time, **kwargs):

        self.time = time
        super().__init__(**kwargs)

    @property
    def time(self):
        """Time (s) used for all computations

        Returns
        -------
        numpy.ndarray
            Time (or times) in seconds used for all computations
        """
        return self._time

    @time.setter
    def time(self, value):

        # Ensure float or numpy array of float
        try:
            value = np.atleast_1d(value).astype(float)
        except:
            raise TypeError(f"times are not a valid type")

        # Enforce positivity and dimensions
        if (value < 0.).any():
            raise ValueError("All times must be greater than 0")
        if value.ndim > 1:
            raise TypeError(f"times must be ('*') array")

        self._time = value

    def peak_time(self, z):
        r"""Compute peak time for a plane wave in a homogeneous electromagnetic media.

        For a particular distance along the propagation path of a planewave in
        a homogeneous media, the `peak time <https://em.geosci.xyz/content/maxwell1_fundamentals/transient_planewaves_homogeneous/peaktime.html>`_
        is the time at which the maximum signal amplitude is observed. In other words,
        it is simple computation for when the peak of the planewave will arrive.

        For the quasistatic case (no electric displacement), the peak time is given by:

        .. math::
            t_{max} = \frac{\mu \sigma z^2}{6}

        where :math:`\mu` is the magnetic permeability, :math:`\sigma` is the electrical
        conductivity and *z* is the propagation distance.

        Parameters
        ----------
        z : float or numpy.ndarray
            propagation distance from the planewave source (m)

        Returns
        -------
        float, np.ndarray
            Peak time/times in seconds

        """
        return peak_time(z, self.sigma, self.mu)

    @property
    def diffusion_distance(self):
        r"""Compute diffusion distance for a plane wave in a homogeneous electromagnetic media.

        For a planewave source in a homogeneous media, the `diffusion distance <https://em.geosci.xyz/content/maxwell1_fundamentals/transient_planewaves_homogeneous/peakdistance.html>`_
        is the propagation distance the peak has travelled at a given time. It is sometimes
        referred to as the peak distance. The diffusion distance is a simple computation for
        how far a planewave has travelled after a certain time.

        For the quasistatic case (no electric displacement), the diffusion distance is given by:

        .. math::
            D = \sqrt{\frac{2 t}{\mu \sigma}}

        where :math:`\mu` is the magnetic permeability, :math:`\sigma` is the electrical
        conductivity and *t* is the time.

        Returns
        -------
        float, np.ndarray
            Diffusion distance at all times.

        """
        return diffusion_distance(self.time, self.sigma, self.mu)

    @property
    def theta(self):
        r"""
        Analog to wavenumber in the frequency domain. See Ward and Hohmann, 1988
        pages 174-175:

        .. math::
            \theta = \sqrt{\frac{\mu \sigma}{4t}}

        Parameters
        ----------
        time : float or numpy.ndarray
            propagation time (s)
        sigma : float or numpy.ndarray
            electrical conductivity (S/m)
        mu : float or numpy.ndarray, optional
            magnetic permeability (A/m). Default is the permeability of free-space
            (:math:`\mu_0`)

        Returns
        -------
        float, np.ndarray
            The theta value for all times.

        """
        return theta(self.time, self.sigma, self.mu)
