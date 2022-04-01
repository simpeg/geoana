import numpy as np
from geoana.em.fdem.base import BaseFDEM
from geoana.spatial import repeat_scalar
from geoana.em.base import BaseElectricDipole, BaseMagneticDipole


class ElectricDipoleWholeSpace(BaseFDEM, BaseElectricDipole):
    r"""Class for simulating the fields and fluxes for a harmonic electric dipole in a wholespace.

    Harmonic electric dipole in a whole space. The source is
    (c.f. Ward and Hohmann, 1988 page 173). The source current
    density for a dipole located at :math:`\mathbf{r}_s` with orientation
    :math:`\mathbf{\hat{u}}`

    .. math::
        \mathbf{J}(\mathbf{r}) = I ds \delta(\mathbf{r} - \mathbf{r}_s)\mathbf{\hat{u}}

    """
    def vector_potential(self, xyz):
        r"""Vector potential for the harmonic current dipole at a set of gridded locations.

        For an electric current dipole oriented in the :math:`\hat{u}` direction with
        dipole moment :math:`I ds`, the magnetic vector potential at frequency :math:`f`
        at vector distance :math:`\mathbf{r}` from the dipole is given by:

        .. math::
            \mathbf{a}(\mathbf{r}) = \frac{I ds}{4 \pi r} e^{-ikr} \mathbf{\hat{u}}

        where

        .. math::
            k = \sqrt{\omega^2 \mu \varepsilon - i \omega \mu \sigma}

        Parameters
        ----------
        xyz : (n, 3) numpy.ndarray
            Gridded xyz locations

        Returns
        -------
        (n_freq, n_loc, 3) numpy.array of complex
            Magnetic vector potential at all frequencies for the gridded
            locations provided. Output array is squeezed when n_freq and/or
            n_loc = 1.

        Examples
        --------
        Here, we define an z-oriented electric dipole and plot the magnetic
        vector potential on the xy-plane that intercepts z=0.

        >>> from geoana.em.fdem import ElectricDipoleWholeSpace
        >>> from geoana.utils import ndgrid
        >>> from geoana.plotting_utils import plot2Ddata
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt

        Let us begin by defining the electric current dipole.

        >>> frequency = np.logspace(1, 3, 3)
        >>> location = np.r_[0., 0., 0.]
        >>> orientation = np.r_[0., 0., 1.]
        >>> current = 1.
        >>> sigma = 1.0
        >>> simulation = ElectricDipoleWholeSpace(
        >>>     frequency, location=location, orientation=orientation,
        >>>     current=current, sigma=sigma
        >>> )

        Now we create a set of gridded locations and compute the vector potential.

        >>> xyz = ndgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20), np.array([0]))
        >>> a = simulation.vector_potential(xyz)

        Finally, we plot the real and imaginary components of the vector potential.
        Given the symmetry, there are only vertical components.

        >>> f_ind = 1
        >>> fig = plt.figure(figsize=(6, 3))
        >>> ax1 = fig.add_axes([0.15, 0.15, 0.40, 0.75])
        >>> plot2Ddata(xyz[:, 0:2], np.real(a[f_ind, :, 2]), ax=ax1, scale='log', ncontour=25)
        >>> ax1.set_xlabel('X')
        >>> ax1.set_ylabel('Z')
        >>> ax1.autoscale(tight=True)
        >>> ax1.set_title('Real component {} Hz'.format(frequency[f_ind]))
        >>> ax2 = fig.add_axes([0.6, 0.15, 0.40, 0.75])
        >>> plot2Ddata(xyz[:, 0:2], np.imag(a[f_ind, :, 2]), ax=ax2, scale='log', ncontour=25)
        >>> ax2.set_xlabel('X')
        >>> ax2.set_yticks([])
        >>> ax2.set_title('Imag component {} Hz'.format(frequency[f_ind]))
        >>> ax2.autoscale(tight=True)

        """
        # r = self.distance(xyz)
        # a = (
        #     (self.current * self.length) / (4*np.pi*r) *
        #     np.exp(-1j*self.wavenumber*r)
        # )
        # a = np.kron(np.ones(1, 3), np.atleast_2d(a).T)
        # return self.dot_orientation(a)

        n_freq = len(self.frequency)
        n_loc = np.shape(xyz)[0]

        r = self.distance(xyz)
        k = self.wavenumber

        # (n_freq, n_loc) array
        a = self.current * self.length * (
            1 / (4*np.pi*np.tile(r.reshape((1, n_loc)), (n_freq, 1))) *
            np.exp(-1j*np.outer(k, r))
        )

        v = self.orientation.reshape(1, 1, 3)
        a = a.reshape((n_freq, n_loc, 1))
        return np.kron(v, a).squeeze()

    def electric_field(self, xyz):
        r"""Electric field for the harmonic current dipole at a set of gridded locations.

        For an electric current dipole oriented in the :math:`\hat{u}` direction with
        dipole moment :math:`I ds` and harmonic frequency :math:`f`, this method computes the
        electric field at the set of gridded xyz locations provided.

        The analytic solution is adapted from Ward and Hohmann (1988). For a harmonic electric
        current dipole oriented in the :math:`\hat{x}` direction, the solution at vector distance
        :math:`\mathbf{r}` from the current dipole is:

        .. math::
            \;\;\;\;\;\; \mathbf{E}(\mathbf{r}) = \frac{I ds}{4\pi (\sigma + i \omega \varepsilon) r^3} & e^{-ikr} ... \\
            & \Bigg [ \Bigg ( \frac{x^2}{r^2}\hat{x} + \frac{xy}{r^2}\hat{y} + \frac{xz}{r^2}\hat{z} \Bigg ) \big ( -k^2r^2 + 3ikr + 3 \big ) + \big ( k^2 r^2 - ikr - 1 \big ) \hat{x} \Bigg ]

        where

        .. math::
            k = \sqrt{\omega^2 \mu \varepsilon - i \omega \mu \sigma}

        Parameters
        ----------
        xyz : (n, 3) numpy.ndarray
            Gridded xyz locations

        Returns
        -------
        (n_freq, n_loc, 3) numpy.array of complex
            Electric field at all frequencies for the gridded
            locations provided. Output array is squeezed when n_freq and/or
            n_loc = 1.

        Examples
        --------
        Here, we define an x-oriented electric dipole and plot the electric
        field on the xz-plane that intercepts y=0.

        >>> from geoana.em.fdem import ElectricDipoleWholeSpace
        >>> from geoana.utils import ndgrid
        >>> from geoana.plotting_utils import plot2Ddata
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt

        Let us begin by defining the electric current dipole.

        >>> frequency = np.logspace(1, 3, 3)
        >>> location = np.r_[0., 0., 0.]
        >>> orientation = np.r_[1., 0., 0.]
        >>> current = 1.
        >>> sigma = 1.0
        >>> simulation = ElectricDipoleWholeSpace(
        >>>     frequency, location=location, orientation=orientation,
        >>>     current=current, sigma=sigma
        >>> )

        Now we create a set of gridded locations and compute the electric field.

        >>> xyz = ndgrid(np.linspace(-1, 1, 20), np.array([0]), np.linspace(-1, 1, 20))
        >>> E = simulation.electric_field(xyz)

        Finally, we plot the real and imaginary components of the electric field

        >>> f_ind = 1
        >>> fig = plt.figure(figsize=(6, 3))
        >>> ax1 = fig.add_axes([0.15, 0.15, 0.40, 0.75])
        >>> plot2Ddata(
        >>>     xyz[:, 0::2], np.real(E[f_ind, :, 0::2]), vec=True, ax=ax1, scale='log', ncontour=25
        >>> )
        >>> ax1.set_xlabel('X')
        >>> ax1.set_ylabel('Z')
        >>> ax1.autoscale(tight=True)
        >>> ax1.set_title('Real component {} Hz'.format(frequency[f_ind]))
        >>> ax2 = fig.add_axes([0.6, 0.15, 0.40, 0.75])
        >>> plot2Ddata(
        >>>     xyz[:, 0::2], np.imag(E[f_ind, :, 0::2]), vec=True, ax=ax2, scale='log', ncontour=25
        >>> )
        >>> ax2.set_xlabel('X')
        >>> ax2.set_yticks([])
        >>> ax2.autoscale(tight=True)
        >>> ax2.set_title('Imag component {} Hz'.format(frequency[f_ind]))

        """
        # dxyz = self.vector_distance(xyz)
        # r = repeat_scalar(self.distance(xyz))
        # kr = self.wavenumber * r
        # ikr = 1j * kr

        # front_term = (
        #     (self.current * self.length) / (4 * np.pi * self.sigma * r**3) *
        #     np.exp(-ikr)
        # )
        # symmetric_term = (
        #     repeat_scalar(self.dot_orientation(dxyz)) * dxyz *
        #     (-kr**2 + 3*ikr + 3) / r**2
        # )
        # oriented_term = (
        #     (kr**2 - ikr - 1) *
        #     np.kron(self.orientation, np.ones((dxyz.shape[0], 1)))
        # )
        # return front_term * (symmetric_term + oriented_term)

        n_freq = len(self.frequency)
        n_loc = np.shape(xyz)[0]

        k = self.wavenumber
        r = self.distance(xyz)
        dxyz = self.vector_distance(xyz)

        # (n_freq, n_loc) arrays
        kr = np.outer(k, r)
        ikr = 1j * kr
        tile_r = np.outer(np.ones(n_freq), r)

        front_term = (self.current * self.length) * (
            1 / (4 * np.pi * self.sigma * tile_r**3) * np.exp(-ikr)
        ).reshape((n_freq, n_loc, 1))
        front_term = np.tile(front_term, (1, 1, 3))

        temp_1 = repeat_scalar(self.dot_orientation(dxyz)) * dxyz
        temp_1 = np.tile(temp_1.reshape((1, n_loc, 3)), (n_freq, 1, 1))
        temp_2 = (-kr**2 + 3*ikr + 3) / tile_r**2
        temp_2 = np.tile(temp_2.reshape((n_freq, n_loc, 1)), (1, 1, 3))
        symmetric_term = temp_1 * temp_2

        temp_1 = (kr**2 - ikr - 1)
        temp_1 = np.tile(temp_1.reshape((n_freq, n_loc, 1)), (1, 1, 3))
        temp_2 = np.kron(self.orientation, np.ones((dxyz.shape[0], 1)))
        temp_2 = np.tile(temp_2.reshape((1, n_loc, 3)), (n_freq, 1, 1))
        oriented_term = temp_1 * temp_2

        return (front_term * (symmetric_term + oriented_term)).squeeze()

    def current_density(self, xyz):
        r"""Current density for the harmonic current dipole at a set of gridded locations.

        For an electric current dipole oriented in the :math:`\hat{u}` direction with
        dipole moment :math:`I ds` and harmonic frequency :math:`f`, this method computes the
        current density at the set of gridded xyz locations provided.

        The analytic solution is adapted from Ward and Hohmann (1988). For a harmonic electric
        current dipole oriented in the :math:`\hat{x}` direction, the solution at vector distance
        :math:`\mathbf{r}` from the current dipole is:

        .. math::
            \;\;\;\; \mathbf{J}(\mathbf{r}) = \frac{\sigma I ds}{4\pi (\sigma + i \omega \varepsilon) r^3} & e^{-ikr} ... \\
            & \Bigg [ \Bigg ( \frac{x^2}{r^2}\hat{x} + \frac{xy}{r^2}\hat{y} + \frac{xz}{r^2}\hat{z} \Bigg ) \big ( -k^2r^2 + 3ikr + 3 \big ) \big ( k^2 r^2 - ikr - 1 \big ) \hat{x} \Bigg ]

        where

        .. math::
            k = \sqrt{\omega^2 \mu \varepsilon - i \omega \mu \sigma}

        Parameters
        ----------
        xyz : (n, 3) numpy.ndarray
            Gridded xyz locations

        Returns
        -------
        (n_freq, n_loc, 3) numpy.array of complex
            Current density at all frequencies for the gridded
            locations provided. Output array is squeezed when n_freq and/or
            n_loc = 1.

        Examples
        --------
        Here, we define an x-oriented electric dipole and plot the current
        density on the xz-plane that intercepts y=0.

        >>> from geoana.em.fdem import ElectricDipoleWholeSpace
        >>> from geoana.utils import ndgrid
        >>> from geoana.plotting_utils import plot2Ddata
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt

        Let us begin by defining the electric current dipole.

        >>> frequency = np.logspace(1, 3, 3)
        >>> location = np.r_[0., 0., 0.]
        >>> orientation = np.r_[1., 0., 0.]
        >>> current = 1.
        >>> sigma = 1.0
        >>> simulation = ElectricDipoleWholeSpace(
        >>>     frequency, location=location, orientation=orientation,
        >>>     current=current, sigma=sigma
        >>> )

        Now we create a set of gridded locations and compute the current density.

        >>> xyz = ndgrid(np.linspace(-1, 1, 20), np.array([0]), np.linspace(-1, 1, 20))
        >>> J = simulation.current_density(xyz)

        Finally, we plot the real and imaginary components of the current density.

        >>> f_ind = 1
        >>> fig = plt.figure(figsize=(6, 3))
        >>> ax1 = fig.add_axes([0.15, 0.15, 0.40, 0.75])
        >>> plot2Ddata(
        >>>     xyz[:, 0::2], np.real(J[f_ind, :, 0::2]), vec=True, ax=ax1, scale='log', ncontour=25
        >>> )
        >>> ax1.set_xlabel('X')
        >>> ax1.set_ylabel('Z')
        >>> ax1.autoscale(tight=True)
        >>> ax1.set_title('Real component {} Hz'.format(frequency[f_ind]))
        >>> ax2 = fig.add_axes([0.6, 0.15, 0.40, 0.75])
        >>> plot2Ddata(
        >>>     xyz[:, 0::2], np.imag(J[f_ind, :, 0::2]), vec=True, ax=ax2, scale='log', ncontour=25
        >>> )
        >>> ax2.set_xlabel('X')
        >>> ax2.set_yticks([])
        >>> ax2.autoscale(tight=True)
        >>> ax2.set_title('Imag component {} Hz'.format(frequency[f_ind]))

        """
        return self.sigma * self.electric_field(xyz)

    def magnetic_field(self, xyz):
        r"""Magnetic field produced by the harmonic current dipole at a set of gridded locations.

        For an electric current dipole oriented in the :math:`\hat{u}` direction with
        dipole moment :math:`I ds` and harmonic frequency :math:`f`, this method computes the
        magnetic field at the set of gridded xyz locations provided.

        The analytic solution is adapted from Ward and Hohmann (1988). For a harmonic electric
        current dipole oriented in the :math:`\hat{x}` direction, the solution at vector distance
        :math:`\mathbf{r}` from the current dipole is:

        .. math::
            \mathbf{H}(\mathbf{r}) = \frac{I ds}{4\pi r^2} (ikr + 1) e^{-ikr} \big ( - \frac{z}{r}\hat{y} + \frac{y}{r}\hat{z} \big )

        where

        .. math::
            k = \sqrt{\omega^2 \mu \varepsilon - i \omega \mu \sigma}

        Parameters
        ----------
        xyz : (n, 3) numpy.ndarray
            Gridded xyz locations

        Returns
        -------
        (n_freq, n_loc, 3) numpy.array of complex
            Magnetic field at all frequencies for the gridded
            locations provided. Output array is squeezed when n_freq and/or
            n_loc = 1.

        Examples
        --------
        Here, we define an z-oriented electric dipole and plot the magnetic field
        on the xy-plane that intercepts z=0.

        >>> from geoana.em.fdem import ElectricDipoleWholeSpace
        >>> from geoana.utils import ndgrid
        >>> from geoana.plotting_utils import plot2Ddata
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt

        Let us begin by defining the electric current dipole.

        >>> frequency = np.logspace(1, 3, 3)
        >>> location = np.r_[0., 0., 0.]
        >>> orientation = np.r_[0., 0., 1.]
        >>> current = 1.
        >>> sigma = 1.0
        >>> simulation = ElectricDipoleWholeSpace(
        >>>     frequency, location=location, orientation=orientation,
        >>>     current=current, sigma=sigma
        >>> )

        Now we create a set of gridded locations and compute the magnetic field.

        >>> xyz = ndgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20), np.array([0]))
        >>> H = simulation.magnetic_field(xyz)

        Finally, we plot the real and imaginary components of the magnetic field.

        >>> f_ind = 1
        >>> fig = plt.figure(figsize=(6, 3))
        >>> ax1 = fig.add_axes([0.15, 0.15, 0.40, 0.75])
        >>> plot2Ddata(
        >>>     xyz[:, 0:2], np.real(H[f_ind, :, 0:2]), vec=True, ax=ax1, scale='log', ncontour=25
        >>> )
        >>> ax1.set_xlabel('X')
        >>> ax1.set_ylabel('Y')
        >>> ax1.autoscale(tight=True)
        >>> ax1.set_title('Real component {} Hz'.format(frequency[f_ind]))
        >>> ax2 = fig.add_axes([0.6, 0.15, 0.40, 0.75])
        >>> plot2Ddata(
        >>>     xyz[:, 0:2], np.imag(H[f_ind, :, 0:2]), vec=True, ax=ax2, scale='log', ncontour=25
        >>> )
        >>> ax2.set_xlabel('X')
        >>> ax2.set_yticks([])
        >>> ax2.autoscale(tight=True)
        >>> ax2.set_title('Imag component {} Hz'.format(frequency[f_ind]))

        """
        # dxyz = self.vector_distance(xyz)
        # r = repeat_scalar(self.distance(xyz))
        # kr = self.wavenumber * r
        # ikr = 1j*kr

        # front_term = (
        #     self.current * self.length / (4 * np.pi * r**2) * (ikr + 1) *
        #     np.exp(-ikr)
        # )
        # return -front_term * self.cross_orientation(dxyz) / r

        n_freq = len(self.frequency)
        n_loc = np.shape(xyz)[0]

        k = self.wavenumber
        r = self.distance(xyz)

        # (n_freq, n_loc)
        kr = np.outer(k, r)
        ikr = 1j * kr
        tile_r = np.outer(np.ones(n_freq), r)

        r = repeat_scalar(r)
        dxyz = self.vector_distance(xyz)

        first_term = self.current * self.length * (
            1 / (4 * np.pi * tile_r**2) * (ikr + 1) * np.exp(-ikr)
        ).reshape((n_freq, n_loc, 1))
        first_term = np.tile(first_term, (1, 1, 3))

        second_term = (self.cross_orientation(dxyz) / r).reshape((1, n_loc, 3))
        second_term = np.tile(second_term, (n_freq, 1, 1))

        return -(first_term * second_term).squeeze()

    def magnetic_flux_density(self, xyz):
        r"""Magnetic flux density produced by the harmonic electric current dipole at a set of gridded locations.

        For an electric current dipole oriented in the :math:`\hat{u}` direction with
        dipole moment :math:`I ds` and harmonic frequency :math:`f`, this method computes the
        magnetic flux density at the set of gridded xyz locations provided.

        The analytic solution is adapted from Ward and Hohmann (1988). For a harmonic electric
        current dipole oriented in the :math:`\hat{x}` direction, the solution at vector distance
        :math:`\mathbf{r}` from the current dipole is:

        .. math::
            \mathbf{B}(\mathbf{r}) = \frac{\mu I ds}{4\pi r^2} (ikr + 1) e^{-ikr} \bigg ( - \frac{z}{r}\hat{y} + \frac{y}{r}\hat{z} \bigg )

        where

        .. math::
            k = \sqrt{\omega^2 \mu \varepsilon - i \omega \mu \sigma}

        Parameters
        ----------
        xyz : (n, 3) numpy.ndarray
            Gridded xyz locations

        Returns
        -------
        (n_freq, n_loc, 3) numpy.array of complex
            Magnetic flux at all frequencies for the gridded
            locations provided. Output array is squeezed when n_freq and/or
            n_loc = 1.

        Examples
        --------
        Here, we define an z-oriented electric dipole and plot the magnetic flux density
        on the xy-plane that intercepts z=0.

        >>> from geoana.em.fdem import ElectricDipoleWholeSpace
        >>> from geoana.utils import ndgrid
        >>> from geoana.plotting_utils import plot2Ddata
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt

        Let us begin by defining the electric current dipole.

        >>> frequency = np.logspace(1, 3, 3)
        >>> location = np.r_[0., 0., 0.]
        >>> orientation = np.r_[0., 0., 1.]
        >>> current = 1.
        >>> sigma = 1.0
        >>> simulation = ElectricDipoleWholeSpace(
        >>>     frequency, location=location, orientation=orientation,
        >>>     current=current, sigma=sigma
        >>> )

        Now we create a set of gridded locations and compute the magnetic flux density.

        >>> xyz = ndgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20), np.array([0]))
        >>> B = simulation.magnetic_flux_density(xyz)

        Finally, we plot the real and imaginary components of the magnetic flux density.

        >>> f_ind = 1
        >>> fig = plt.figure(figsize=(6, 3))
        >>> ax1 = fig.add_axes([0.15, 0.15, 0.40, 0.75])
        >>> plot2Ddata(
        >>>     xyz[:, 0:2], np.real(B[f_ind, :, 0:2]), vec=True, ax=ax1, scale='log', ncontour=25
        >>> )
        >>> ax1.set_xlabel('X')
        >>> ax1.set_ylabel('Y')
        >>> ax1.autoscale(tight=True)
        >>> ax1.set_title('Real component {} Hz'.format(frequency[f_ind]))
        >>> ax2 = fig.add_axes([0.6, 0.15, 0.40, 0.75])
        >>> plot2Ddata(
        >>>     xyz[:, 0:2], np.imag(B[f_ind, :, 0:2]), vec=True, ax=ax2, scale='log', ncontour=25
        >>> )
        >>> ax2.set_xlabel('X')
        >>> ax2.set_yticks([])
        >>> ax2.autoscale(tight=True)
        >>> ax2.set_title('Imag component {} Hz'.format(frequency[f_ind]))

        """
        return self.mu * self.magnetic_field(xyz)


class MagneticDipoleWholeSpace(BaseFDEM, BaseMagneticDipole):
    """
    Harmonic magnetic dipole in a whole space.
    """

    def vector_potential(self, xyz):
        r"""Vector potential for the harmonic magnetic dipole at a set of gridded locations.

        For a harmonic magnetic dipole oriented in the :math:`\hat{u}` direction with
        moment amplitude :math:`m`, the magnetic vector potential at frequency :math:`f`
        at vector distance :math:`\mathbf{r}` from the dipole is given by:

        .. math::
            \mathbf{a}(\mathbf{r}) = \frac{i \omega \mu m}{4 \pi r} e^{-ikr}
            \mathbf{\hat{u}}

        where

        .. math::
            k = \sqrt{\omega^2 \mu \varepsilon - i \omega \mu \sigma}

        Parameters
        ----------
        xyz : (n, 3) numpy.ndarray
            Gridded xyz locations

        Returns
        -------
        (n_freq, n_loc, 3) numpy.array of complex
            Magnetic vector potential at all frequencies for the gridded
            locations provided. Output array is squeezed when n_freq and/or
            n_loc = 1.

        Examples
        --------
        Here, we define an z-oriented magnetic dipole and plot the magnetic
        vector potential on the xy-plane that intercepts z=0.

        >>> from geoana.em.fdem import MagneticDipoleWholeSpace
        >>> from geoana.utils import ndgrid
        >>> from geoana.plotting_utils import plot2Ddata
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt

        Let us begin by defining the electric current dipole.

        >>> frequency = np.logspace(1, 3, 3)
        >>> location = np.r_[0., 0., 0.]
        >>> orientation = np.r_[0., 0., 1.]
        >>> moment = 1.
        >>> sigma = 1.0
        >>> simulation = MagneticDipoleWholeSpace(
        >>>     frequency, location=location, orientation=orientation,
        >>>     moment=moment, sigma=sigma
        >>> )

        Now we create a set of gridded locations and compute the vector potential.

        >>> xyz = ndgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20), np.array([0]))
        >>> a = simulation.vector_potential(xyz)

        Finally, we plot the real and imaginary components of the vector potential.

        >>> f_ind = 1
        >>> fig = plt.figure(figsize=(6, 3))
        >>> ax1 = fig.add_axes([0.15, 0.15, 0.40, 0.75])
        >>> plot2Ddata(xyz[:, 0:2], np.real(a[f_ind, :, 2]), ax=ax1, scale='log', ncontour=25)
        >>> ax1.set_xlabel('X')
        >>> ax1.set_ylabel('Z')
        >>> ax1.autoscale(tight=True)
        >>> ax1.set_title('Real component {} Hz'.format(frequency[f_ind]))
        >>> ax2 = fig.add_axes([0.6, 0.15, 0.40, 0.75])
        >>> plot2Ddata(xyz[:, 0:2], np.imag(a[f_ind, :, 2]), ax=ax2, scale='log', ncontour=25)
        >>> ax2.set_xlabel('X')
        >>> ax2.set_yticks([])
        >>> ax2.autoscale(tight=True)
        >>> ax2.set_title('Imag component {} Hz'.format(frequency[f_ind]))

        """
        # r = self.distance(xyz)
        # f = (
        #     (1j * self.omega * self.mu * self.moment) / (4 * np.pi * r) *
        #     np.exp(-1j * self.wavenumber * r)
        # )
        # f = np.kron(np.ones(1, 3), np.atleast_2d(f).T)
        # return self.dot_orientation(f)

        n_freq = len(self.frequency)
        n_loc = np.shape(xyz)[0]

        r = self.distance(xyz)
        k = self.wavenumber

        tile_r = np.tile(r.reshape((1, n_loc)), (n_freq, 1))
        tile_w = np.tile(self.omega.reshape((n_freq, 1)), (1, n_loc))

        a = (1j * tile_w * self.mu * self.moment) * (
            1 / (4*np.pi*tile_r) * np.exp(-1j*np.outer(k, r))
        )

        v = self.orientation.reshape(1, 1, 3)
        a = a.reshape((n_freq, n_loc, 1))

        return np.kron(v, a).squeeze()

    def electric_field(self, xyz):
        r"""Electric field for the harmonic magnetic dipole at a set of gridded locations.

        For a harmonic magnetic dipole oriented in the :math:`\hat{u}` direction with
        moment amplitude :math:`m` and harmonic frequency :math:`f`, this method computes the
        electric field at the set of gridded xyz locations provided.

        The analytic solution is adapted from Ward and Hohmann (1988). For a harmonic electric
        magnetic dipole oriented in the :math:`\hat{x}` direction, the solution at vector distance
        :math:`\mathbf{r}` from the dipole is:

        .. math::
            \mathbf{E}(\mathbf{r}) = \frac{i\omega \mu m}{4\pi r^2} (ikr + 1) e^{-ikr} \big ( - \frac{z}{r}\hat{y} + \frac{y}{r}\hat{z} \big )

        where

        .. math::
            k = \sqrt{\omega^2 \mu \varepsilon - i \omega \mu \sigma}

        Parameters
        ----------
        xyz : (n, 3) numpy.ndarray
            Gridded xyz locations

        Returns
        -------
        (n_freq, n_loc, 3) numpy.array of complex
            Electric field at all frequencies for the gridded
            locations provided. Output array is squeezed when n_freq and/or
            n_loc = 1.

        Examples
        --------
        Here, we define an z-oriented magnetic dipole and plot the electric
        field on the xy-plane that intercepts z=0.

        >>> from geoana.em.fdem import MagneticDipoleWholeSpace
        >>> from geoana.utils import ndgrid
        >>> from geoana.plotting_utils import plot2Ddata
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt

        Let us begin by defining the electric current dipole.

        >>> frequency = np.logspace(1, 3, 3)
        >>> location = np.r_[0., 0., 0.]
        >>> orientation = np.r_[0., 0., 1.]
        >>> moment = 1.
        >>> sigma = 1.0
        >>> simulation = MagneticDipoleWholeSpace(
        >>>     frequency, location=location, orientation=orientation,
        >>>     moment=moment, sigma=sigma
        >>> )

        Now we create a set of gridded locations and compute the electric field.

        >>> xyz = ndgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20), np.array([0]))
        >>> E = simulation.electric_field(xyz)

        Finally, we plot the real and imaginary components of the electric field

        >>> f_ind = 1
        >>> fig = plt.figure(figsize=(6, 3))
        >>> ax1 = fig.add_axes([0.15, 0.15, 0.40, 0.75])
        >>> plot2Ddata(
        >>>     xyz[:, 0:2], np.real(E[f_ind, :, 0:2]), vec=True, ax=ax1, scale='log', ncontour=25
        >>> )
        >>> ax1.set_xlabel('X')
        >>> ax1.set_ylabel('Y')
        >>> ax1.autoscale(tight=True)
        >>> ax1.set_title('Real component {} Hz'.format(frequency[f_ind]))
        >>> ax2 = fig.add_axes([0.6, 0.15, 0.40, 0.75])
        >>> plot2Ddata(
        >>>     xyz[:, 0:2], np.imag(E[f_ind, :, 0:2]), vec=True, ax=ax2, scale='log', ncontour=25
        >>> )
        >>> ax2.set_xlabel('X')
        >>> ax2.set_yticks([])
        >>> ax2.autoscale(tight=True)
        >>> ax2.set_title('Imag component {} Hz'.format(frequency[f_ind]))

        """
        # dxyz = self.vector_distance(xyz)
        # r = repeat_scalar(self.distance(xyz))
        # kr = self.wavenumber*r
        # ikr = 1j * kr

        # front_term = (
        #     (1j * self.omega * self.mu * self.moment) / (4. * np.pi * r**2) *
        #     (ikr + 1) * np.exp(-ikr)
        # )
        # return front_term * self.cross_orientation(dxyz) / r

        n_freq = len(self.frequency)
        n_loc = np.shape(xyz)[0]

        k = self.wavenumber
        r = self.distance(xyz)

        # (n_freq, n_loc)
        tile_r = np.tile(r.reshape((1, n_loc)), (n_freq, 1))
        tile_w = np.tile(self.omega.reshape((n_freq, 1)), (1, n_loc))
        kr = np.outer(k, r)
        ikr = 1j * kr

        first_term = (
            (1j * tile_w * self.mu * self.moment) *
            (1 / (4 * np.pi * tile_r**2) * (ikr + 1) * np.exp(-ikr))
        ).reshape((n_freq, n_loc, 1))
        first_term = np.tile(first_term, (1, 1, 3))

        r = repeat_scalar(r)
        dxyz = self.vector_distance(xyz)

        second_term = (self.cross_orientation(dxyz) / r).reshape((1, n_loc, 3))
        second_term = np.tile(second_term, (n_freq, 1, 1))

        return (first_term * second_term).squeeze()


    def current_density(self, xyz):
        r"""Current density for the harmonic magnetic dipole at a set of gridded locations.

        For a harmonic magnetic dipole oriented in the :math:`\hat{u}` direction with
        moment amplitude :math:`m` and harmonic frequency :math:`f`, this method computes the
        current density at the set of gridded xyz locations provided.

        The analytic solution is adapted from Ward and Hohmann (1988). For a harmonic electric
        magnetic dipole oriented in the :math:`\hat{x}` direction, the solution at vector distance
        :math:`\mathbf{r}` from the dipole is:

        .. math::
            \mathbf{J}(\mathbf{r}) = \frac{i\omega \mu \sigma m}{4\pi r^2} (ikr + 1) e^{-ikr} \big ( - \frac{z}{r}\hat{y} + \frac{y}{r}\hat{z} \big )

        where

        .. math::
            k = \sqrt{\omega^2 \mu \varepsilon - i \omega \mu \sigma}

        Parameters
        ----------
        xyz : (n, 3) numpy.ndarray
            Gridded xyz locations

        Returns
        -------
        (n_freq, n_loc, 3) numpy.array of complex
            Current density at all frequencies for the gridded
            locations provided. Output array is squeezed when n_freq and/or
            n_loc = 1.

        Examples
        --------
        Here, we define an z-oriented magnetic dipole and plot the electric
        current density on the xy-plane that intercepts z=0.

        >>> from geoana.em.fdem import MagneticDipoleWholeSpace
        >>> from geoana.utils import ndgrid
        >>> from geoana.plotting_utils import plot2Ddata
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt

        Let us begin by defining the electric current dipole.

        >>> frequency = np.logspace(1, 3, 3)
        >>> location = np.r_[0., 0., 0.]
        >>> orientation = np.r_[0., 0., 1.]
        >>> moment = 1.
        >>> sigma = 1.0
        >>> simulation = MagneticDipoleWholeSpace(
        >>>     frequency, location=location, orientation=orientation,
        >>>     moment=moment, sigma=sigma
        >>> )

        Now we create a set of gridded locations and compute the current density.

        >>> xyz = ndgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20), np.array([0]))
        >>> J = simulation.current_density(xyz)

        Finally, we plot the real and imaginary components of the current density.

        >>> f_ind = 2
        >>> fig = plt.figure(figsize=(6, 3))
        >>> ax1 = fig.add_axes([0.15, 0.15, 0.40, 0.75])
        >>> plot2Ddata(
        >>>     xyz[:, 0:2], np.real(J[f_ind, :, 0:2]), vec=True, ax=ax1, scale='log', ncontour=25
        >>> )
        >>> ax1.set_xlabel('X')
        >>> ax1.set_ylabel('Y')
        >>> ax1.autoscale(tight=True)
        >>> ax1.set_title('Real component {} Hz'.format(frequency[f_ind]))
        >>> ax2 = fig.add_axes([0.6, 0.15, 0.40, 0.75])
        >>> plot2Ddata(
        >>>     xyz[:, 0:2], np.imag(J[f_ind, :, 0:2]), vec=True, ax=ax2, scale='log', ncontour=25
        >>> )
        >>> ax2.set_xlabel('X')
        >>> ax2.set_yticks([])
        >>> ax2.autoscale(tight=True)
        >>> ax2.set_title('Imag component {} Hz'.format(frequency[f_ind]))

        """
        return self.sigma * self.electric_field(xyz)

    def magnetic_field(self, xyz):
        r"""Magnetic field for the harmonic magnetic dipole at a set of gridded locations.

        For a harmonic magnetic dipole oriented in the :math:`\hat{u}` direction with
        moment amplitude :math:`m` and harmonic frequency :math:`f`, this method computes the
        magnetic field at the set of gridded xyz locations provided.

        The analytic solution is adapted from Ward and Hohmann (1988). For a harmonic magnetic
        dipole oriented in the :math:`\hat{x}` direction, the solution at vector distance
        :math:`\mathbf{r}` from the dipole is:

        .. math::
            \mathbf{H}(\mathbf{r}) = \frac{m}{4\pi r^3} e^{-ikr} \Bigg [ \Bigg ( \frac{x^2}{r^2}\hat{x} + \frac{xy}{r^2}\hat{y} + \frac{xz}{r^2}\hat{z} \Bigg ) ... \\
            \big ( -k^2r^2 + 3ikr + 3 \big ) \big ( k^2 r^2 - ikr - 1 \big ) \hat{x} \Bigg ]

        where

        .. math::
            k = \sqrt{\omega^2 \mu \varepsilon - i \omega \mu \sigma}

        Parameters
        ----------
        xyz : (n, 3) numpy.ndarray
            Gridded xyz locations

        Returns
        -------
        (n_freq, n_loc, 3) numpy.array of complex
            Magnetic field at all frequencies for the gridded
            locations provided. Output array is squeezed when n_freq and/or
            n_loc = 1.

        Examples
        --------
        Here, we define an z-oriented magnetic dipole and plot the magnetic field
        on the xz-plane that intercepts y=0.

        >>> from geoana.em.fdem import MagneticDipoleWholeSpace
        >>> from geoana.utils import ndgrid
        >>> from geoana.plotting_utils import plot2Ddata
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt

        Let us begin by defining the electric current dipole.

        >>> frequency = np.logspace(1, 3, 3)
        >>> location = np.r_[0., 0., 0.]
        >>> orientation = np.r_[0., 0., 1.]
        >>> moment = 1.
        >>> sigma = 1.0
        >>> simulation = MagneticDipoleWholeSpace(
        >>>     frequency, location=location, orientation=orientation,
        >>>     moment=moment, sigma=sigma
        >>> )

        Now we create a set of gridded locations and compute the magnetic field.

        >>> xyz = ndgrid(np.linspace(-1, 1, 20), np.array([0]), np.linspace(-1, 1, 20))
        >>> H = simulation.magnetic_field(xyz)

        Finally, we plot the real and imaginary components of the magnetic field.

        >>> f_ind = 2
        >>> fig = plt.figure(figsize=(6, 3))
        >>> ax1 = fig.add_axes([0.15, 0.15, 0.40, 0.75])
        >>> plot2Ddata(
        >>>     xyz[:, 0::2], np.real(H[f_ind, :, 0::2]), vec=True, ax=ax1, scale='log', ncontour=25
        >>> )
        >>> ax1.set_xlabel('X')
        >>> ax1.set_ylabel('Z')
        >>> ax1.autoscale(tight=True)
        >>> ax1.set_title('Real component {} Hz'.format(frequency[f_ind]))
        >>> ax2 = fig.add_axes([0.6, 0.15, 0.40, 0.75])
        >>> plot2Ddata(
        >>>     xyz[:, 0::2], np.imag(H[f_ind, :, 0::2]), vec=True, ax=ax2, scale='log', ncontour=25
        >>> )
        >>> ax2.set_xlabel('X')
        >>> ax2.set_yticks([])
        >>> ax2.autoscale(tight=True)
        >>> ax2.set_title('Imag component {} Hz'.format(frequency[f_ind]))

        """
        # dxyz = self.vector_distance(xyz)
        # r = repeat_scalar(self.distance(xyz))
        # kr = self.wavenumber*r
        # ikr = 1j*kr

        # front_term = self.moment / (4. * np.pi * r**3) * np.exp(-ikr)
        # symmetric_term = (
        #     repeat_scalar(self.dot_orientation(dxyz)) * dxyz *
        #     (-kr**2 + 3*ikr + 3) / r**2
        # )
        # oriented_term = (
        #     (kr**2 - ikr - 1) *
        #     np.kron(self.orientation, np.ones((dxyz.shape[0], 1)))
        # )

        # return front_term * (symmetric_term + oriented_term)

        n_freq = len(self.frequency)
        n_loc = np.shape(xyz)[0]

        k = self.wavenumber
        r = self.distance(xyz)
        dxyz = self.vector_distance(xyz)

        # (n_freq, n_loc)
        kr = np.outer(k, r)
        ikr = 1j * kr
        tile_r = np.outer(np.ones(n_freq), r)

        front_term = self.moment * (
            1 / (4 * np.pi * tile_r**3) * np.exp(-ikr)
        ).reshape((n_freq, n_loc, 1))
        front_term = np.tile(front_term, (1, 1, 3))

        temp_1 = repeat_scalar(self.dot_orientation(dxyz)) * dxyz
        temp_1 = np.tile(temp_1.reshape((1, n_loc, 3)), (n_freq, 1, 1))
        temp_2 = (-kr**2 + 3*ikr + 3) / tile_r**2
        temp_2 = np.tile(temp_2.reshape((n_freq, n_loc, 1)), (1, 1, 3))
        symmetric_term = temp_1 * temp_2

        temp_1 = (kr**2 - ikr - 1)
        temp_1 = np.tile(temp_1.reshape((n_freq, n_loc, 1)), (1, 1, 3))
        temp_2 = np.kron(self.orientation, np.ones((dxyz.shape[0], 1)))
        temp_2 = np.tile(temp_2.reshape((1, n_loc, 3)), (n_freq, 1, 1))
        oriented_term = temp_1 * temp_2

        return (front_term * (symmetric_term + oriented_term)).squeeze()

    def magnetic_flux_density(self, xyz):
        r"""Magnetic flux density for the harmonic magnetic dipole at a set of gridded locations.

        For a harmonic magnetic dipole oriented in the :math:`\hat{u}` direction with
        moment amplitude :math:`m` and harmonic frequency :math:`f`, this method computes the
        magnetic flux density at the set of gridded xyz locations provided.

        The analytic solution is adapted from Ward and Hohmann (1988). For a harmonic magnetic
        dipole oriented in the :math:`\hat{x}` direction, the solution at vector distance
        :math:`\mathbf{r}` from the dipole is:

        .. math::
            \mathbf{B}(\mathbf{r}) = \frac{\mu m}{4\pi r^3} e^{-ikr} \Bigg [ \Bigg ( \frac{x^2}{r^2}\hat{x} + \frac{xy}{r^2}\hat{y} + \frac{xz}{r^2}\hat{z} \Bigg ) ... \\
            \big ( -k^2r^2 + 3ikr + 3 \big ) \big ( k^2 r^2 - ikr - 1 \big ) \hat{x} \Bigg ]

        where

        .. math::
            k = \sqrt{\omega^2 \mu \varepsilon - i \omega \mu \sigma}

        Parameters
        ----------
        xyz : (n, 3) numpy.ndarray
            Gridded xyz locations

        Returns
        -------
        (n_freq, n_loc, 3) numpy.array of complex
            Magnetic flux density at all frequencies for the gridded
            locations provided. Output array is squeezed when n_freq and/or
            n_loc = 1.

        Examples
        --------
        Here, we define an z-oriented magnetic dipole and plot the magnetic flux density
        on the xz-plane that intercepts y=0.

        >>> from geoana.em.fdem import MagneticDipoleWholeSpace
        >>> from geoana.utils import ndgrid
        >>> from geoana.plotting_utils import plot2Ddata
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt

        Let us begin by defining the electric current dipole.

        >>> frequency = np.logspace(1, 3, 3)
        >>> location = np.r_[0., 0., 0.]
        >>> orientation = np.r_[0., 0., 1.]
        >>> moment = 1.
        >>> sigma = 1.0
        >>> simulation = MagneticDipoleWholeSpace(
        >>>     frequency, location=location, orientation=orientation,
        >>>     moment=moment, sigma=sigma
        >>> )

        Now we create a set of gridded locations and compute the magnetic flux density.

        >>> xyz = ndgrid(np.linspace(-1, 1, 20), np.array([0]), np.linspace(-1, 1, 20))
        >>> B = simulation.magnetic_field(xyz)

        Finally, we plot the real and imaginary components of the magnetic flux density.

        >>> f_ind = 1
        >>> fig = plt.figure(figsize=(6, 3))
        >>> ax1 = fig.add_axes([0.15, 0.15, 0.40, 0.75])
        >>> plot2Ddata(
        >>>     xyz[:, 0::2], np.real(B[f_ind, :, 0::2]), vec=True, ax=ax1, scale='log', ncontour=25
        >>> )
        >>> ax1.set_xlabel('X')
        >>> ax1.set_ylabel('Z')
        >>> ax1.autoscale(tight=True)
        >>> ax1.set_title('Real component {} Hz'.format(frequency[f_ind]))
        >>> ax2 = fig.add_axes([0.6, 0.15, 0.40, 0.75])
        >>> plot2Ddata(
        >>>     xyz[:, 0::2], np.imag(B[f_ind, :, 0::2]), vec=True, ax=ax2, scale='log', ncontour=25
        >>> )
        >>> ax2.set_xlabel('X')
        >>> ax2.set_yticks([])
        >>> ax2.autoscale(tight=True)
        >>> ax2.set_title('Imag component {} Hz'.format(frequency[f_ind]))

        """
        return self.mu * self.magnetic_field(xyz)
