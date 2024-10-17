from scipy.special import erf
import numpy as np

from geoana.em.tdem.base import BaseTDEM
from geoana.spatial import repeat_scalar
from geoana.utils import check_xyz_dim, append_ndim
from geoana.em.base import BaseElectricDipole

###############################################################################
#                                                                             #
#                                  Classes                                    #
#                                                                             #
###############################################################################


class ElectricDipoleWholeSpace(BaseTDEM, BaseElectricDipole):
    r"""
    Class for simulating the fields and fluxes for a transient electric
    dipole in a wholespace.

    """

    def vector_potential(self, xyz):
        r"""Vector potential for the transient current dipole at a set of gridded locations.

        For an electric current dipole oriented in the :math:`\hat{u}`
        direction with dipole moment :math:`I ds`, this method computes
        the transient vector potential at the set of observation times for
        the gridded xyz locations provided.

        The analytic solution is adapted from Ward and Hohmann (1988).
        For a transient electric current dipole oriented in the
        :math:`\hat{x}` direction, the solution at vector distance
        :math:`\mathbf{r}` from the current dipole is:

        .. math::
            \mathbf{a}(t) = \frac{I ds}{4 \pi r} \, erf(\theta r) \, \hat{x}

        where

        .. math::
            \theta = \Bigg ( \frac{\mu\sigma}{4t} \Bigg )^{1/2}

        Parameters
        ----------
        xyz : (..., 3) numpy.ndarray
            Gridded xyz locations

        Returns
        -------
        (n_time, ..., 3) numpy.ndarray of float
            Vector potential at all times for the gridded
            locations provided.

        Examples
        --------
        Here, we define an x-oriented electric dipole and plot the vector
        potential on the xz-plane that intercepts y=0.

        >>> from geoana.em.tdem import ElectricDipoleWholeSpace
        >>> from geoana.utils import ndgrid
        >>> from geoana.plotting_utils import plot2Ddata
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt

        Let us begin by defining the electric current dipole.

        >>> time = np.logspace(-6, -2, 3)
        >>> location = np.r_[0., 0., 0.]
        >>> orientation = np.r_[1., 0., 0.]
        >>> current = 1.
        >>> sigma = 1.0
        >>> simulation = ElectricDipoleWholeSpace(
        >>>     time, location=location, orientation=orientation,
        >>>     current=current, sigma=sigma
        >>> )

        Now we create a set of gridded locations and compute the vector potential.

        >>> xyz = ndgrid(np.linspace(-10, 10, 20), np.array([0]), np.linspace(-10, 10, 20))
        >>> E = simulation.vector_potential(xyz)

        Finally, we plot the vector potential at the desired locations/times.

        >>> t_ind = 0
        >>> fig = plt.figure(figsize=(4, 4))
        >>> ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
        >>> plot2Ddata(xyz[:, 0::2], E[t_ind, :, 0::2], ax=ax, vec=True, scale='log')
        >>> ax.set_xlabel('X')
        >>> ax.set_ylabel('Z')
        >>> ax.set_title('Vector potential at {} s'.format(time[t_ind]))
        """
        xyz = check_xyz_dim(xyz)

        r_vec = xyz - self.location
        r = np.linalg.norm(r_vec, axis=-1, keepdims=True)
        theta = append_ndim(self.theta, r.ndim)

        theta_r = theta * r

        return self.current * self.length / (4 * np.pi * r) * erf(theta_r) * self.orientation

    def electric_field(self, xyz):
        r"""Electric field for the transient current dipole at a set of gridded locations.

        For an electric current dipole oriented in the :math:`\hat{u}`
        direction with dipole moment :math:`I ds`, this method computes
        the transient electric field at the set of observation times for
        the gridded xyz locations provided.

        The analytic solution is adapted from Ward and Hohmann (1988).
        For a transient electric current dipole oriented in the
        :math:`\hat{x}` direction, the solution at vector distance
        :math:`\mathbf{r}` from the current dipole is:

        .. math::
            {\bf e_e}(t) = \frac{Ids}{4\pi \sigma r^3} & \Bigg [ \Bigg ( \frac{x^2}{r^2}\mathbf{\hat x} + \frac{xy}{r^2}\mathbf{\hat y} + \frac{xz}{r^2}\mathbf{\hat z} \Bigg ) ... \\
            & \Bigg ( 3 \, \textrm{erf}(\theta r) - \bigg ( \frac{4}{\sqrt{\pi}}\theta^3 r^3 + \frac{6}{\sqrt{\pi}} \theta r \bigg ) e^{-\theta^2 r^2}  \Bigg ) - \Bigg ( \textrm{erf}(\theta r) - \bigg ( \frac{4}{\sqrt{\pi}} \theta^3 r^3 + \frac{2}{\sqrt{\pi}} \theta r \bigg ) e^{-\theta^2 r^2} \Bigg ) \mathbf{\hat x} \Bigg ]

        where

        .. math::
            \theta = \Bigg ( \frac{\mu\sigma}{4t} \Bigg )^{1/2}

        Parameters
        ----------
        xyz : (..., 3) numpy.ndarray
            Gridded xyz locations

        Returns
        -------
        (n_time, ..., 3) numpy.ndarray of float
            Electric field at all times for the gridded
            locations provided.

        Examples
        --------
        Here, we define an x-oriented electric dipole and plot the electric
        field on the xz-plane that intercepts y=0.

        >>> from geoana.em.tdem import ElectricDipoleWholeSpace
        >>> from geoana.utils import ndgrid
        >>> from geoana.plotting_utils import plot2Ddata
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt

        Let us begin by defining the electric current dipole.

        >>> time = np.logspace(-6, -2, 3)
        >>> location = np.r_[0., 0., 0.]
        >>> orientation = np.r_[1., 0., 0.]
        >>> current = 1.
        >>> sigma = 1.0
        >>> simulation = ElectricDipoleWholeSpace(
        >>>     time, location=location, orientation=orientation,
        >>>     current=current, sigma=sigma
        >>> )

        Now we create a set of gridded locations and compute the electric field.

        >>> xyz = ndgrid(np.linspace(-10, 10, 20), np.array([0]), np.linspace(-10, 10, 20))
        >>> E = simulation.electric_field(xyz)

        Finally, we plot the electric field at the desired locations/times.

        >>> t_ind = 0
        >>> fig = plt.figure(figsize=(4, 4))
        >>> ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
        >>> plot2Ddata(xyz[:, 0::2], E[t_ind, :, 0::2], ax=ax, vec=True, scale='log')
        >>> ax.set_xlabel('X')
        >>> ax.set_ylabel('Z')
        >>> ax.set_title('Electric field at {} s'.format(time[t_ind]))

        """
        xyz = check_xyz_dim(xyz)

        r_vec = xyz - self.location
        r = np.linalg.norm(r_vec, axis=-1, keepdims=True)
        r_hat = r_vec / r
        theta = append_ndim(self.theta, r.ndim)
        theta_r = theta * r

        root_pi = np.sqrt(np.pi)

        front = self.current * self.length / (4 * np.pi * self.sigma * r**3)

        common_part = theta_r / root_pi * np.exp(-theta_r**2)

        sym_term = 3 * erf(theta_r) - (4 * theta_r**2 + 6) * common_part
        sym_direction = r_hat.dot(self.orientation)[..., None] * r_hat

        orient_term = erf(theta_r) - (4 * theta_r**2 + 2) * common_part
        orient_dir = self.orientation

        return front * (sym_term * sym_direction - orient_term * orient_dir)

    def current_density(self, xyz):
        r"""Current density for the transient current dipole at a set of gridded locations.

        For an electric current dipole oriented in the :math:`\hat{u}`
        direction with dipole moment :math:`I ds`, this method computes
        the transient current density at the set of observation times for
        the gridded xyz locations provided.

        The analytic solution is adapted from Ward and Hohmann (1988).
        For a transient electric current dipole oriented in the
        :math:`\hat{x}` direction, the solution at vector distance
        :math:`\mathbf{r}` from the current dipole is:

        .. math::
            {\bf e_e}(t) = \frac{Ids}{4\pi \sigma r^2} & \Bigg [ \Bigg ( \frac{x^2}{r^2}\mathbf{\hat x} + \frac{xy}{r^2}\mathbf{\hat y} + \frac{xz}{r^2}\mathbf{\hat z} \Bigg ) \Bigg ( 3 \, \textrm{erf}(\theta r) - \bigg ( \frac{4}{\sqrt{\pi}}\theta^3 r^3 \; ... \\
            & + \frac{6}{\sqrt{\pi}} \theta r \bigg ) e^{-\theta^2 r^2}  \Bigg ) - \Bigg ( \textrm{erf}(\theta r) - \bigg ( \frac{4}{\sqrt{\pi}} \theta^3 r^3 + \frac{2}{\sqrt{\pi}} \theta r \bigg ) e^{-\theta^2 r^2} \Bigg ) \mathbf{\hat x} \Bigg ]

        where

        .. math::
            \theta = \Bigg ( \frac{\mu\sigma}{4t} \Bigg )^{1/2}

        Parameters
        ----------
        xyz : (..., 3) numpy.ndarray
            Gridded xyz locations

        Returns
        -------
        (n_time, ..., 3) numpy.ndarray of float
            Current density at all times for the gridded
            locations provided.

        Examples
        --------
        Here, we define an x-oriented electric dipole and plot the current density
        on the xz-plane that intercepts y=0.

        >>> from geoana.em.tdem import ElectricDipoleWholeSpace
        >>> from geoana.utils import ndgrid
        >>> from geoana.plotting_utils import plot2Ddata
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt

        Let us begin by defining the electric current dipole.

        >>> time = np.logspace(-6, -2, 3)
        >>> location = np.r_[0., 0., 0.]
        >>> orientation = np.r_[1., 0., 0.]
        >>> current = 1.
        >>> sigma = 1.0
        >>> simulation = ElectricDipoleWholeSpace(
        >>>     time, location=location, orientation=orientation,
        >>>     current=current, sigma=sigma
        >>> )

        Now we create a set of gridded locations and compute the current density.

        >>> xyz = ndgrid(np.linspace(-10, 10, 20), np.array([0]), np.linspace(-10, 10, 20))
        >>> J = simulation.current_density(xyz)

        Finally, we plot the current density at the desired locations/times.

        >>> t_ind = 0
        >>> fig = plt.figure(figsize=(4, 4))
        >>> ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
        >>> plot2Ddata(xyz[:, 0::2], J[t_ind, :, 0::2], ax=ax, vec=True, scale='log')
        >>> ax.set_xlabel('X')
        >>> ax.set_ylabel('Z')
        >>> ax.set_title('Current density at {} s'.format(time[t_ind]))

        """
        return self.sigma * self.electric_field(xyz)

    def magnetic_field(self, xyz):
        r"""Magnetic field for the transient current dipole at a set of gridded locations.

        For an electric current dipole oriented in the :math:`\hat{u}`
        direction with dipole moment :math:`I ds`, this method computes
        the transient magnetic field at the set of observation times for
        the gridded xyz locations provided.

        The analytic solution is adapted from Ward and Hohmann (1988).
        For a transient electric current dipole oriented in the
        :math:`\hat{x}` direction, the solution at vector distance
        :math:`\mathbf{r}` from the current dipole is:

        .. math::
            \mathbf{h}(t) = \frac{Ids}{4 \pi r^3} \bigg ( \textrm{erf}(\theta r) - \frac{2}{\sqrt{\pi}} \theta r \, e^{-\theta^2 r^2}  \bigg ) \big ( - z \, \mathbf{\hat y} + y \, \mathbf{\hat z}  \big )

        where

        .. math::
            \theta = \Bigg ( \frac{\mu\sigma}{4t} \Bigg )^{1/2}

        Parameters
        ----------
        xyz : (..., 3) numpy.ndarray
            Gridded xyz locations

        Returns
        -------
        (n_time, ..., 3) numpy.ndarray of float
            Transient magnetic field at all times for the gridded
            locations provided.

        Examples
        --------
        Here, we define a z-oriented electric dipole and plot the magnetic field
        on the xy-plane that intercepts z=0.

        >>> from geoana.em.tdem import ElectricDipoleWholeSpace
        >>> from geoana.utils import ndgrid
        >>> from geoana.plotting_utils import plot2Ddata
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt

        Let us begin by defining the electric current dipole.

        >>> time = np.logspace(-6, -2, 3)
        >>> location = np.r_[0., 0., 0.]
        >>> orientation = np.r_[0., 0., 1.]
        >>> current = 1.
        >>> sigma = 1.0
        >>> simulation = ElectricDipoleWholeSpace(
        >>>     time, location=location, orientation=orientation,
        >>>     current=current, sigma=sigma
        >>> )

        Now we create a set of gridded locations and compute the magnetic field.

        >>> xyz = ndgrid(np.linspace(-10, 10, 20), np.linspace(-10, 10, 20), np.array([0]))
        >>> H = simulation.magnetic_field(xyz)

        Finally, we plot the magnetic field at the desired locations/times.

        >>> t_ind = 0
        >>> fig = plt.figure(figsize=(4, 4))
        >>> ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
        >>> plot2Ddata(xyz[:, 0:2], H[t_ind, :, 0:2], ax=ax, vec=True, scale='log')
        >>> ax.set_xlabel('X')
        >>> ax.set_ylabel('Y')
        >>> ax.set_title('Magnetic field at {} s'.format(time[t_ind]))

        """
        xyz = check_xyz_dim(xyz)

        r_vec = xyz - self.location
        r = np.linalg.norm(r_vec, axis=-1, keepdims=True)
        r_hat = r_vec / r
        theta = append_ndim(self.theta, r.ndim)

        theta_r = theta * r

        term1 = (self.current * self.length) / (4 * np.pi * r**2)
        term2 = erf(theta_r) - 2 / np.sqrt(np.pi) * theta_r * np.exp(-theta_r**2)

        direction = np.cross(self.orientation, r_hat)

        return term1 * term2 * direction

    def magnetic_field_time_deriv(self, xyz):
        r"""Time-derivative of the magnetic field for the transient current dipole at a set of gridded locations.

        For an electric current dipole oriented in the :math:`\hat{u}`
        direction with dipole moment :math:`I ds`, this method computes
        the time-derivative of the transient magnetic field at the set of observation times for
        the gridded xyz locations provided.

        The analytic solution is adapted from Ward and Hohmann (1988).
        For a transient electric current dipole oriented in the
        :math:`\hat{x}` direction, the solution at vector distance
        :math:`\mathbf{r}` from the current dipole is:

        .. math::
            \frac{\partial \mathbf{h}}{\partial t} = - \frac{2 \, \theta^5 Ids}{\pi^{3/2} \mu \sigma} e^{-\theta^2 r^2} \big ( - z \, \mathbf{\hat y} + y \, \mathbf{\hat z}  \big )

        where

        .. math::
            \theta = \Bigg ( \frac{\mu\sigma}{4t} \Bigg )^{1/2}

        Parameters
        ----------
        xyz : (..., 3) numpy.ndarray
            Gridded xyz locations

        Returns
        -------
        (n_time, ..., 3) numpy.ndarray of float
            Time-derivative of the transient magnetic field at all times for the gridded
            locations provided.

        Examples
        --------
        Here, we define a z-oriented electric dipole and plot the time-derivative
        of the magnetic field on the xy-plane that intercepts z=0.

        >>> from geoana.em.tdem import ElectricDipoleWholeSpace
        >>> from geoana.utils import ndgrid
        >>> from geoana.plotting_utils import plot2Ddata
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt

        Let us begin by defining the electric current dipole.

        >>> time = np.logspace(-6, -2, 3)
        >>> location = np.r_[0., 0., 0.]
        >>> orientation = np.r_[0., 0., 1.]
        >>> current = 1.
        >>> sigma = 1.0
        >>> simulation = ElectricDipoleWholeSpace(
        >>>     time, location=location, orientation=orientation,
        >>>     current=current, sigma=sigma
        >>> )

        Now we create a set of gridded locations and compute the dh/dt.

        >>> xyz = ndgrid(np.linspace(-10, 10, 20), np.linspace(-10, 10, 20), np.array([0]))
        >>> dHdt = simulation.magnetic_field_time_deriv(xyz)

        Finally, we plot dH/dt at the desired locations/times.

        >>> t_ind = 0
        >>> fig = plt.figure(figsize=(4, 4))
        >>> ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
        >>> plot2Ddata(xyz[:, 0:2], dHdt[t_ind, :, 0:2], ax=ax, vec=True, scale='log')
        >>> ax.set_xlabel('X')
        >>> ax.set_ylabel('Y')
        >>> ax.set_title('dH/dt at {} s'.format(time[t_ind]))

        """
        xyz = check_xyz_dim(xyz)

        r_vec = xyz - self.location
        r = np.linalg.norm(r_vec, axis=-1, keepdims=True)
        r_hat = r_vec / r
        theta = append_ndim(self.theta, r.ndim)
        time = append_ndim(self.time, r.ndim)

        theta_r = theta * r

        term1 = (self.current * self.length) / (4 * np.pi * r**2)
        term2_dt = -2 * theta_r**3 / (np.sqrt(np.pi) * time) * np.exp(-theta_r**2)

        direction = np.cross(self.orientation, r_hat)

        return term1 * term2_dt * direction

    def magnetic_flux_density(self, xyz):
        r"""Magnetic flux density for the transient current dipole at a set of gridded locations.

        For an electric current dipole oriented in the :math:`\hat{u}`
        direction with dipole moment :math:`I ds`, this method computes
        the transient magnetic flux density at the set of observation times for
        the gridded xyz locations provided.

        The analytic solution is adapted from Ward and Hohmann (1988).
        For a transient electric current dipole oriented in the
        :math:`\hat{x}` direction, the solution at vector distance
        :math:`\mathbf{r}` from the current dipole is:

        .. math::
            \mathbf{b}(t) = \frac{\mu Ids}{4 \pi r^3} \bigg ( \textrm{erf}(\theta r) - \frac{2}{\sqrt{\pi}} \theta r \, e^{-\theta^2 r^2}  \bigg ) \big ( - z \, \mathbf{\hat y} + y \, \mathbf{\hat z}  \big )

        where

        .. math::
            \theta = \Bigg ( \frac{\mu\sigma}{4t} \Bigg )^{1/2}

        Parameters
        ----------
        xyz : (..., 3) numpy.ndarray
            Gridded xyz locations

        Returns
        -------
        (n_time, ..., 3) numpy.ndarray of float
            Transient magnetic field at all times for the gridded
            locations provided.

        Examples
        --------
        Here, we define a z-oriented electric dipole and plot the magnetic flux density
        on the xy-plane that intercepts z=0.

        >>> from geoana.em.tdem import ElectricDipoleWholeSpace
        >>> from geoana.utils import ndgrid
        >>> from geoana.plotting_utils import plot2Ddata
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt

        Let us begin by defining the electric current dipole.

        >>> time = np.logspace(-6, -2, 3)
        >>> location = np.r_[0., 0., 0.]
        >>> orientation = np.r_[0., 0., 1.]
        >>> current = 1.
        >>> sigma = 1.0
        >>> simulation = ElectricDipoleWholeSpace(
        >>>     time, location=location, orientation=orientation,
        >>>     current=current, sigma=sigma
        >>> )

        Now we create a set of gridded locations and compute the magnetic flux density.

        >>> xyz = ndgrid(np.linspace(-10, 10, 20), np.linspace(-10, 10, 20), np.array([0]))
        >>> B = simulation.magnetic_flux_density(xyz)

        Finally, we plot the magnetic flux density at the desired locations/times.

        >>> t_ind = 0
        >>> fig = plt.figure(figsize=(4, 4))
        >>> ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
        >>> plot2Ddata(xyz[:, 0:2], B[t_ind, :, 0:2], ax=ax, vec=True, scale='log')
        >>> ax.set_xlabel('X')
        >>> ax.set_ylabel('Y')
        >>> ax.set_title('Magnetic flux density at {} s'.format(time[t_ind]))

        """

        return self.mu * self.magnetic_field(xyz)

    def magnetic_flux_density_time_deriv(self, xyz):
        r"""Time-derivative of the magnetic flux density for the transient current dipole at a set of gridded locations.

        For a transient electric current dipole oriented in the :math:`\hat{u}`
        direction with dipole moment :math:`I ds`, this method computes
        the time-derivative of the time-derivative of the magnetic flux density at the set of observation times for
        the gridded xyz locations provided.

        The analytic solution is adapted from Ward and Hohmann (1988).
        For a transient electric current dipole oriented in the
        :math:`\hat{x}` direction, the solution at vector distance
        :math:`\mathbf{r}` from the current dipole is:

        .. math::
            \frac{\partial \mathbf{b}}{\partial t} = - \frac{2 \, \theta^5 Ids}{\pi^{3/2} \sigma} e^{-\theta^2 r^2} \big ( - z \, \mathbf{\hat y} + y \, \mathbf{\hat z}  \big )

        where

        .. math::
            \theta = \Bigg ( \frac{\mu\sigma}{4t} \Bigg )^{1/2}

        Parameters
        ----------
        xyz : (..., 3) numpy.ndarray
            Gridded xyz locations

        Returns
        -------
        (n_time, ..., 3) numpy.ndarray of float
            Time-derivative of the transient magnetic flux density at all times for the gridded
            locations provided.

        Examples
        --------
        Here, we define a z-oriented electric dipole and plot the time-derivative of
        the magnetic flux density on the xy-plane that intercepts z=0.

        >>> from geoana.em.tdem import ElectricDipoleWholeSpace
        >>> from geoana.utils import ndgrid
        >>> from geoana.plotting_utils import plot2Ddata
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt

        Let us begin by defining the electric current dipole.

        >>> time = np.logspace(-6, -2, 3)
        >>> location = np.r_[0., 0., 0.]
        >>> orientation = np.r_[0., 0., 1.]
        >>> current = 1.
        >>> sigma = 1.0
        >>> simulation = ElectricDipoleWholeSpace(
        >>>     time, location=location, orientation=orientation,
        >>>     current=current, sigma=sigma
        >>> )

        Now we create a set of gridded locations and compute dB/dt.

        >>> xyz = ndgrid(np.linspace(-10, 10, 20), np.linspace(-10, 10, 20), np.array([0]))
        >>> dBdt = simulation.magnetic_flux_density_time_deriv(xyz)

        Finally, we plot the dB/dt at the desired locations/times.

        >>> t_ind = 0
        >>> fig = plt.figure(figsize=(4, 4))
        >>> ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
        >>> plot2Ddata(xyz[:, 0:2], dBdt[t_ind, :, 0:2], ax=ax, vec=True, scale='log')
        >>> ax.set_xlabel('X')
        >>> ax.set_ylabel('Y')
        >>> ax.set_title('dBdt at {} s'.format(time[t_ind]))

        """

        return self.mu * self.magnetic_field_time_deriv(xyz)


class TransientPlaneWave(BaseTDEM):
    """
    Class for simulating the fields for a transient planewave in a wholespace.
    The wave is assumed to be propogating vertically downward, starting at time=0
    and z = 0

    Parameters
    ----------
    amplitude : float
        amplitude of primary electric field.  Default is 1
    orientation : (3,) array_like or {'X','Y'}
        Orientation of the planewave. Can be defined using as an ``array_like`` of length 3,
        or by using one of {'X','Y'} to define a planewave along the x or y direction.
        Default is 'X'.
    """

    def __init__(
        self, amplitude=1.0, orientation='X', **kwargs
    ):

        self.amplitude = amplitude
        self.orientation = orientation
        super().__init__(**kwargs)

    @property
    def amplitude(self):
        """Amplitude of the primary field.

        Returns
        -------
        float
            Amplitude of the primary field. Default = 1
        """
        return self._amplitude

    @amplitude.setter
    def amplitude(self, item):

        item = float(item)
        self._amplitude = item

    @property
    def orientation(self):
        """Orientation of the planewave as a normalized vector

        Returns
        -------
        (3) numpy.ndarray of float or str in {'X','Y'}
            planewave orientation, normalized to unit magnitude
        """
        return self._orientation

    @orientation.setter
    def orientation(self, var):

        if isinstance(var, str):
            if var.upper() == 'X':
                var = np.r_[1., 0., 0.]
            elif var.upper() == 'Y':
                var = np.r_[0., 1., 0.]
        else:
            try:
                var = np.asarray(var, dtype=float)
            except:
                raise TypeError(
                    f"orientation must be str or array_like, got {type(var)}"
                )
            var = np.squeeze(var)
            if var.shape != (3,):
                raise ValueError(
                    f"orientation must be array_like with shape (3,), got {len(var)}"
                )
            if var[2] != 0:
                raise ValueError(
                    f"z axis of orientation must be 0 in order to stay in the xy-plane, got {var[2]}"
                )

            # Normalize the orientation
            var /= np.linalg.norm(var)

        self._orientation = var

    def electric_field(self, xyz):
        r"""Electric field for the transient planewave at a set of gridded locations.

        Parameters
        ----------
        xyz : (..., 3) numpy.ndarray
            Gridded xyz locations

        Returns
        -------
        (n_t, ..., 3) numpy.ndarray of float
            Electric field at all times for the gridded
            locations provided.

        Examples
        --------
        Here, we define a transient planewave in the x-direction in a wholespace.

        >>> from geoana.em.tdem import TransientPlaneWave
        >>> import numpy as np
        >>> from geoana.utils import ndgrid
        >>> from mpl_toolkits.axes_grid1 import make_axes_locatable
        >>> import matplotlib.pyplot as plt

        Let us begin by defining the transient planewave in the x-direction.

        >>> time = 1.0
        >>> orientation = 'X'
        >>> sigma = 1.0
        >>> simulation = TransientPlaneWave(
        >>>     time=time, orientation=orientation, sigma=sigma
        >>> )

        Now we create a set of gridded locations and compute the electric field.

        >>> x = np.linspace(-1, 1, 20)
        >>> z = np.linspace(-1000, 0, 20)
        >>> xyz = ndgrid(x, np.array([0]), z)
        >>> e_vec = simulation.electric_field(xyz)
        >>> ex = e_vec[..., 0]

        Finally, we plot the x-oriented electric field.

        >>> plt.pcolor(x, z, ex.reshape(20, 20), shading='auto')
        >>> cb = plt.colorbar()
        >>> cb.set_label(label= 'Electric Field ($V/m$)')
        >>> plt.ylabel('Z coordinate ($m$)')
        >>> plt.xlabel('X coordinate ($m$)')
        >>> plt.title('Electric Field of a Transient Planewave in the x-direction in a Wholespace')
        >>> plt.show()
        """
        xyz = check_xyz_dim(xyz)
        e0 = self.amplitude

        z = xyz[..., [2]]
        t = append_ndim(self.time, xyz.ndim)

        mu = self.mu
        sigma = self.sigma

        bunja = -e0 * (mu * sigma) ** 0.5 * z * np.exp(-(mu * sigma * z ** 2) / (4 * t))
        bunmo = 2 * np.pi ** 0.5 * t ** 1.5

        return self.orientation * (bunja / bunmo)

    def current_density(self, xyz):
        r"""Current density for the transient planewave at a set of gridded locations.

        Parameters
        ----------
        xyz : (..., 3) numpy.ndarray
            Gridded xyz locations

        Returns
        -------
        (n_t, ..., 3) numpy.ndarray of float
            Current density at all frequencies for the gridded
            locations provided.

        Examples
        --------
        Here, we define a transient planewave in the x-direction in a wholespace.

        >>> from geoana.em.tdem import TransientPlaneWave
        >>> import numpy as np
        >>> from geoana.utils import ndgrid
        >>> from mpl_toolkits.axes_grid1 import make_axes_locatable
        >>> import matplotlib.pyplot as plt

        Let us begin by defining the transient planewave in the x-direction.

        >>> time = 1.0
        >>> orientation = 'X'
        >>> sigma = 1.0
        >>> simulation = TransientPlaneWave(
        >>>     time=time, orientation=orientation, sigma=sigma
        >>> )

        Now we create a set of gridded locations and compute the electric field.

        >>> x = np.linspace(-1, 1, 20)
        >>> z = np.linspace(-1000, 0, 20)
        >>> xyz = ndgrid(x, np.array([0]), z)
        >>> j_vec = simulation.current_density(xyz)
        >>> jx = j_vec[..., 0]

        Finally, we plot the x-oriented current density.

        >>> plt.pcolor(x, z, jx.reshape(20, 20), shading='auto')
        >>> cb = plt.colorbar()
        >>> cb.set_label(label= 'Current Density ($A/m^2$)')
        >>> plt.ylabel('Z coordinate ($m$)')
        >>> plt.xlabel('X coordinate ($m$)')
        >>> plt.title('Current Density of a Transient Planewave in the x-direction in a Wholespace')
        >>> plt.show()
        """
        return self.sigma * self.electric_field(xyz)

    def magnetic_field(self, xyz):
        r"""Magnetic field for the harmonic planewave at a set of gridded locations.

        Parameters
        ----------
        xyz : (...,, 3) numpy.ndarray
            Gridded xyz locations

        Returns
        -------
        (n_t, ..., 3) numpy.ndarray of float
            Magnetic field at all frequencies for the gridded
            locations provided.

        Examples
        --------
        Here, we define a transient planewave in the x-direction in a wholespace.

        >>> from geoana.em.tdem import TransientPlaneWave
        >>> import numpy as np
        >>> from geoana.utils import ndgrid
        >>> from mpl_toolkits.axes_grid1 import make_axes_locatable
        >>> import matplotlib.pyplot as plt

        Let us begin by defining the transient planewave in the x-direction.

        >>> time = 1.0
        >>> orientation = 'X'
        >>> sigma = 1.0
        >>> simulation = TransientPlaneWave(
        >>>     time=time, orientation=orientation, sigma=sigma
        >>> )

        Now we create a set of gridded locations and compute the electric field.

        >>> x = np.linspace(-1, 1, 20)
        >>> z = np.linspace(-1000, 0, 20)
        >>> xyz = ndgrid(x, np.array([0]), z)
        >>> h_vec = simulation.magnetic_field(xyz)
        >>> hy = h_vec[..., 1]

        Finally, we plot the x-oriented magnetic field.

        >>> plt.pcolor(x, z, hy.reshape(20, 20), shading='auto')
        >>> cb = plt.colorbar()
        >>> cb.set_label(label= 'Magnetic Field ($A/m$)')
        >>> plt.ylabel('Z coordinate ($m$)')
        >>> plt.xlabel('X coordinate ($m$)')
        >>> plt.title('Magnetic Field of a Transient Planewave in the x-direction in a Wholespace')
        >>> plt.show()
        """
        return self.magnetic_flux_density(xyz) / self.mu

    def magnetic_flux_density(self, xyz):
        r"""Magnetic flux density for the transient planewave at a set of gridded locations.

        Parameters
        ----------
        xyz : (..., 3) numpy.ndarray
            Gridded xyz locations

        Returns
        -------
        (n_t, ..., 3) numpy.ndarray of float
            Magnetic flux density at all frequencies for the gridded
            locations provided.

        Examples
        --------
        Here, we define a transient planewave in the x-direction in a wholespace.

        >>> from geoana.em.tdem import TransientPlaneWave
        >>> import numpy as np
        >>> from geoana.utils import ndgrid
        >>> from mpl_toolkits.axes_grid1 import make_axes_locatable
        >>> import matplotlib.pyplot as plt

        Let us begin by defining the transient planewave in the x-direction.

        >>> time = 1.0
        >>> orientation = 'X'
        >>> sigma = 1.0
        >>> simulation = TransientPlaneWave(
        >>>     time=time, orientation=orientation, sigma=sigma
        >>> )

        Now we create a set of gridded locations and compute the magnetic flux density.

        >>> x = np.linspace(-1, 1, 20)
        >>> z = np.linspace(-1000, 0, 20)
        >>> xyz = ndgrid(x, np.array([0]), z)
        >>> b_vec = simulation.magnetic_flux_density(xyz)
        >>> by = b_vec[..., 1]

        Finally, we plot the x-oriented magnetic flux density.

        >>> plt.pcolor(x, z, by.reshape(20, 20), shading='auto')
        >>> cb = plt.colorbar()
        >>> cb.set_label(label= 'Magnetic Flux Density (T)')
        >>> plt.ylabel('Z coordinate ($m$)')
        >>> plt.xlabel('X coordinate ($m$)')
        >>> plt.title('Magnetic Flux Density of a Transient Planewave in the x-direction in a Wholespace')
        >>> plt.show()
        """
        xyz = check_xyz_dim(xyz)

        e0 = self.amplitude

        z = xyz[..., [2]]
        t = append_ndim(self.time, xyz.ndim)

        sigma = self.sigma
        mu = self.mu

        # Curl E = -dB/dt
        b_amp = - e0 * np.sqrt(sigma * mu / (np.pi * t)) * np.exp((-mu * sigma * z ** 2)/(4 * t))

        # account for the orientation in the cross product
        # take cross product with the propagation direction
        b_dir = np.cross(self.orientation, [0, 0, -1])

        return b_dir * b_amp
