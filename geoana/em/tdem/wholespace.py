from scipy.special import erf
import numpy as np
from geoana.em.tdem.base import BaseTDEM
from geoana.spatial import repeat_scalar
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
        xyz : (n, 3) numpy.ndarray
            Gridded xyz locations

        Returns
        -------
        (n_time, n_loc, 3) numpy.array of float
            Vector potential at all times for the gridded
            locations provided. Output array is squeezed when n_time and/or
            n_loc = 1.

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

        r = self.distance(xyz)

        n_loc = len(r)
        n_time = len(self.time)

        theta_r = np.outer(self.theta, r)
        tile_r = np.outer(np.ones(n_time), r)

        term_1 = (
            (self.current * self.length) * erf(theta_r) / (4 * np.pi * tile_r**3)
        ).reshape((n_time, n_loc, 1))
        term_1 = np.tile(term_1, (1, 1, 3))

        term_2 = np.tile(np.reshape(self.orientation, (1, 1, 3)), (n_time, n_loc, 1))

        return (term_1 * term_2).squeeze()

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
        xyz : (n, 3) numpy.ndarray
            Gridded xyz locations

        Returns
        -------
        (n_time, n_loc, 3) numpy.array of float
            Electric field at all times for the gridded
            locations provided. Output array is squeezed when n_time and/or
            n_loc = 1.

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
        # dxyz = self.vector_distance(xyz)
        # r = self.distance(xyz)
        # r = repeat_scalar(r)
        # theta_r = self.theta * r
        # root_pi = np.sqrt(np.pi)

        # front = (
        #     (self.current * self.length) / (4 * np.pi * self.sigma * r**3)
        # )

        # symmetric_term = (
        #     (
        #         - (
        #             4/root_pi * theta_r ** 3 + 6/root_pi * theta_r
        #         ) * np.exp(-theta_r**2) +
        #         3 * erf(theta_r)
        #     ) * (
        #         repeat_scalar(self.dot_orientation(dxyz)) * dxyz / r**2
        #     )
        # )

        # oriented_term = (
        #     (
        #         4./root_pi * theta_r**3 + 2./root_pi * theta_r
        #     ) * np.exp(-theta_r**2) -
        #     erf(theta_r)
        # ) * np.kron(self.orientation, np.ones((dxyz.shape[0], 1)))

        # return front * (symmetric_term + oriented_term)

        root_pi = np.sqrt(np.pi)
        dxyz = self.vector_distance(xyz)
        r = self.distance(xyz)

        n_loc = len(r)
        n_time = len(self.time)

        theta_r = np.outer(self.theta, r)
        tile_r = np.outer(np.ones(n_time), r)
        r = repeat_scalar(r)

        front = (
            (self.current * self.length) / (4 * np.pi * self.sigma * tile_r**3)
        ).reshape((n_time, n_loc, 1))
        front = np.tile(front, (1, 1, 3))

        term_1 = 3 * erf(theta_r) - (4/root_pi * theta_r ** 3 + 6/root_pi * theta_r) * np.exp(-theta_r**2)
        term_1 = np.tile(term_1.reshape((n_time, n_loc, 1)), (1, 1, 3))
        term_2 = repeat_scalar(self.dot_orientation(dxyz)) * dxyz / r**2
        term_2 = np.tile(term_2.reshape((1, n_loc, 3)), (n_time, 1, 1))
        symmetric_term = term_1 * term_2

        term_1 = (4./root_pi * theta_r**3 + 2./root_pi * theta_r) * np.exp(-theta_r**2) - erf(theta_r)
        term_1 = np.tile(term_1.reshape((n_time, n_loc, 1)), (1, 1, 3))
        term_2 = np.kron(self.orientation, np.ones((dxyz.shape[0], 1)))
        term_2 = np.tile(term_2.reshape((1, n_loc, 3)), (n_time, 1, 1))
        oriented_term = term_1 * term_2

        return (front * (symmetric_term + oriented_term)).squeeze()

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
        xyz : (n, 3) numpy.ndarray
            Gridded xyz locations

        Returns
        -------
        (n_time, n_loc, 3) numpy.array of float
            Current density at all times for the gridded
            locations provided. Output array is squeezed when n_time and/or
            n_loc = 1.

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
        xyz : (n, 3) numpy.ndarray
            Gridded xyz locations

        Returns
        -------
        (n_time, n_loc, 3) numpy.array of float
            Transient magnetic field at all times for the gridded
            locations provided. Output array is squeezed when n_time and/or
            n_loc = 1.

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
        # dxyz = self.vector_distance(xyz)
        # r = self.distance(dxyz)
        # r = repeat_scalar(r)
        # thetar = self.theta * r

        # front = (
        #     self.current * self.length / (4 * np.pi * r**2) * (
        #         2 / np.sqrt(np.pi) * thetar * np.exp(-thetar**2) + erf(thetar)
        #     )
        # )

        # return - front * self.cross_orientation(xyz) / r

        dxyz = self.vector_distance(xyz)
        r = self.distance(dxyz)

        n_loc = len(r)
        n_time = len(self.time)

        theta_r = np.outer(self.theta, r)
        tile_r = np.outer(np.ones(n_time), r)

        term_1 = (
            self.current * self.length / (4 * np.pi * tile_r**2) * (
                2 / np.sqrt(np.pi) * theta_r * np.exp(-theta_r**2) + erf(theta_r)
            )
        ).reshape((n_time, n_loc, 1))

        r = repeat_scalar(r)
        term_2 = (self.cross_orientation(xyz) / r).reshape((1, n_loc, 3))

        return - (np.tile(term_1, (1, 1, 3)) * np.tile(term_2, (n_time, 1, 1))).squeeze()

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
        xyz : (n, 3) numpy.ndarray
            Gridded xyz locations

        Returns
        -------
        (n_time, n_loc, 3) numpy.array of float
            Time-derivative of the transient magnetic field at all times for the gridded
            locations provided. Output array is squeezed when n_time and/or
            n_loc = 1.

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
        # dxyz = self.vector_distance(xyz)
        # r = self.distance(dxyz)
        # r = repeat_scalar(r)

        # front = (
        #     self.current * self.length * self.theta**3 * r /
        #     (2 * np.sqrt(np.pi)**3 * self.time)
        # )

        # return - front * self.cross_orientation(xyz) / r

        dxyz = self.vector_distance(xyz)
        r = self.distance(dxyz)

        n_loc = len(r)
        n_time = len(self.time)

        theta3_r = np.outer(self.theta**3, r)
        tile_t = np.outer(self.time, np.ones(n_loc))

        term_1 = (
            self.current * self.length * theta3_r /
            (2 * np.sqrt(np.pi)**3 * tile_t)
        ).reshape((n_time, n_loc, 1))

        r = repeat_scalar(r)
        term_2 = (self.cross_orientation(xyz) / r).reshape((1, n_loc, 3))

        return - (np.tile(term_1, (1, 1, 3)) * np.tile(term_2, (n_time, 1, 1))).squeeze()

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
        xyz : (n, 3) numpy.ndarray
            Gridded xyz locations

        Returns
        -------
        (n_time, n_loc, 3) numpy.array of float
            Transient magnetic field at all times for the gridded
            locations provided. Output array is squeezed when n_time and/or
            n_loc = 1.

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
        xyz : (n, 3) numpy.ndarray
            Gridded xyz locations

        Returns
        -------
        (n_time, n_loc, 3) numpy.array of float
            Time-derivative of the transient magnetic flux density at all times for the gridded
            locations provided. Output array is squeezed when n_time and/or
            n_loc = 1.

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
