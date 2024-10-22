import numpy as np
from geoana.em.fdem.base import BaseFDEM
from geoana.utils import check_xyz_dim, append_ndim
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
        xyz : (..., 3) numpy.ndarray
            Gridded xyz locations

        Returns
        -------
        (n_freq, ..., 3) numpy.ndarray of complex
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
        xyz = check_xyz_dim(xyz)

        r_vec = xyz - self.location
        r = np.linalg.norm(r_vec, axis=-1, keepdims=True)

        k = append_ndim(self.wavenumber, r.ndim)

        a = self.current * self.length / (4*np.pi*r) * np.exp(-1j*k * r)
        return a * self.orientation

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
        xyz : (..., 3) numpy.ndarray
            Gridded xyz locations

        Returns
        -------
        (n_freq, ..., 3) numpy.ndarray of complex
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
        xyz = check_xyz_dim(xyz)

        r_vec = xyz - self.location
        r = np.linalg.norm(r_vec, axis=-1, keepdims=True)
        r_hat = r_vec / r
        k = append_ndim(self.wavenumber, r.ndim)

        kr = k * r
        ikr = 1j * kr

        front_term = (
            (self.current * self.length) / (4 * np.pi * self.sigma * r**3) *
            np.exp(-ikr)
        )
        symmetric_term = (r_hat @ self.orientation)[..., None] * (-kr**2 + 3*ikr + 3) * r_hat
        oriented_term = (kr**2 - ikr - 1) * self.orientation

        return front_term * (symmetric_term + oriented_term)

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
        xyz : (..., 3) numpy.ndarray
            Gridded xyz locations

        Returns
        -------
        (n_freq, ..., 3) numpy.ndarray of complex
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
        xyz : (..., 3) numpy.ndarray
            Gridded xyz locations

        Returns
        -------
        (n_freq, ..., 3) numpy.ndarray of complex
            Magnetic field at all frequencies for the gridded
            locations provided.

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
        xyz = check_xyz_dim(xyz)

        r_vec = xyz - self.location
        r = np.linalg.norm(r_vec, axis=-1, keepdims=True)
        r_hat = r_vec/r

        k = append_ndim(self.wavenumber, r.ndim)
        kr = k * r
        ikr = 1j*kr

        front_term = (
            self.current * self.length / (4 * np.pi * r**2) * (ikr + 1) *
            np.exp(-ikr)
        )
        return front_term * np.cross(self.orientation, r_hat)

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
        xyz : (..., 3) numpy.ndarray
            Gridded xyz locations

        Returns
        -------
        (n_freq, ..., 3) numpy.ndarray of complex
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
        moment amplitude :math:`m`, the electric vector potential at frequency :math:`f`
        at vector distance :math:`\mathbf{r}` from the dipole is given by:

        .. math::
            \mathbf{F}(\mathbf{r}) = \frac{i \omega \mu m}{4 \pi r} e^{-ikr}
            \mathbf{\hat{u}}

        where

        .. math::
            k = \sqrt{\omega^2 \mu \varepsilon - i \omega \mu \sigma}

        Parameters
        ----------
        xyz : (..., 3) numpy.ndarray
            Gridded xyz locations

        Returns
        -------
        (n_freq, ..., 3) numpy.ndarray of complex
            Magnetic vector potential at all frequencies for the gridded
            locations provided.

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
        xyz = check_xyz_dim(xyz)
        r_vec = xyz - self.location
        r = np.linalg.norm(r_vec, axis=-1, keepdims=True)

        k = append_ndim(self.wavenumber, r.ndim)
        omega = append_ndim(self.omega, r.ndim)

        f = 1j * omega * self.mu * self.moment / (4 * np.pi * r) * np.exp(-1j * k * r)
        return f * self.orientation

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
        xyz : (..., 3) numpy.ndarray
            Gridded xyz locations

        Returns
        -------
        (n_freq, ..., 3) numpy.ndarray of complex
            Electric field at all frequencies for the gridded
            locations provided.

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
        xyz = check_xyz_dim(xyz)
        r_vec = xyz - self.location
        r = np.linalg.norm(r_vec, axis=-1, keepdims=True)
        r_hat = r_vec / r
        k = append_ndim(self.wavenumber, r.ndim)
        omega = append_ndim(self.omega, r.ndim)

        kr = k * r
        ikr = 1j * kr

        front_term = (
             (1j * omega * self.mu * self.moment) / (4. * np.pi * r**2) *
             (ikr + 1) * np.exp(-ikr)
        )
        return front_term * np.cross(r_hat, self.orientation)

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
        xyz : (..., 3) numpy.ndarray
            Gridded xyz locations

        Returns
        -------
        (n_freq, ..., 3) numpy.ndarray of complex
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
        xyz : (..., 3) numpy.ndarray
            Gridded xyz locations

        Returns
        -------
        (n_freq, ..., 3) numpy.ndarray of complex
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
        xyz = check_xyz_dim(xyz)
        r_vec = xyz - self.location
        r = np.linalg.norm(r_vec, axis=-1, keepdims=True)
        r_hat = r_vec / r
        k = append_ndim(self.wavenumber, r.ndim)

        kr = k * r
        ikr = 1j * kr

        front_term = self.moment / (4. * np.pi * r**3) * np.exp(-ikr)

        symmetric_term = (r_hat @ self.orientation)[..., None]*(-kr**2 + 3*ikr + 3) * r_hat

        oriented_term = (kr**2 - ikr - 1) * self.orientation

        return front_term * (symmetric_term + oriented_term)

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
        xyz : (..., 3) numpy.ndarray
            Gridded xyz locations

        Returns
        -------
        (n_freq, ..., 3) numpy.ndarray of complex
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


class HarmonicPlaneWave(BaseFDEM):
    """
    Class for simulating the fields and densities for a harmonic planewave in a wholespace.
    The direction of propogation is assumed to be vertically downwards.

    Parameters
    ----------
    amplitude : float
        amplitude of primary electric field.  Default is 1A.
    orientation : (3) array_like or {'X','Y'}
        Orientation of the planewave. Can be defined using as an ``array_like`` of length 3,
        with z = 0 or by using one of {'X','Y'} to define a planewave along the x or y direction.
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
            Amplitude of the primary field. Default = 1A
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
        r"""Electric field for the harmonic planewave at a set of gridded locations.

        .. math::
            \nabla^2 \mathbf{E} + k^2 \mathbf{E} = 0

        where

        .. math::
            k = \sqrt{\omega^2 \mu \varepsilon - i \omega \mu \sigma}

        Parameters
        ----------
        xyz : (..., 3) numpy.ndarray
            Gridded xyz locations

        Returns
        -------
        (n_f, ..., 3) numpy.ndarray of complex
            Electric field at all frequencies for the gridded
            locations provided.

        Examples
        --------
        Here, we define a harmonic planewave in the x-direction in a wholespace.

        >>> from geoana.em.fdem import HarmonicPlaneWave
        >>> import numpy as np
        >>> from geoana.utils import ndgrid
        >>> from mpl_toolkits.axes_grid1 import make_axes_locatable
        >>> import matplotlib.pyplot as plt

        Let us begin by defining the harmonic planewave in the x-direction.

        >>> frequency = 1
        >>> orientation = 'X'
        >>> sigma = 1.0
        >>> simulation = HarmonicPlaneWave(
        >>>     frequency=frequency, orientation=orientation, sigma=sigma
        >>> )

        Now we create a set of gridded locations and compute the electric field.

        >>> x = np.linspace(-1, 1, 20)
        >>> z = np.linspace(-1000, 0, 20)
        >>> xyz = ndgrid(x, np.array([0]), z)
        >>> e_vec = simulation.electric_field(xyz)
        >>> ex = e_vec[..., 0]
        >>> ey = e_vec[..., 1]
        >>> ez = e_vec[..., 2]

        Finally, we plot the real and imaginary parts of the x-oriented electric field.

        >>> fig, axs = plt.subplots(2, 1, figsize=(14, 12))
        >>> titles = ['Real Part', 'Imaginary Part']
        >>> for ax, V, title in zip(axs.flatten(), [np.real(ex).reshape(20, 20), np.imag(ex).reshape(20, 20)], titles):
        >>>     im = ax.pcolor(x, z, V, shading='auto')
        >>>     divider = make_axes_locatable(ax)
        >>>     cax = divider.append_axes("right", size="5%", pad=0.05)
        >>>     cb = plt.colorbar(im, cax=cax)
        >>>     cb.set_label(label= 'Electric Field ($V/m$)')
        >>>     ax.set_ylabel('Z coordinate ($m$)')
        >>>     ax.set_xlabel('X coordinate ($m$)')
        >>>     ax.set_title(title)
        >>> plt.tight_layout()
        >>> plt.show()
        """
        xyz = check_xyz_dim(xyz)
        z = xyz[..., [2]]

        k = append_ndim(self.wavenumber, xyz.ndim)
        e0 = self.amplitude
        ikz = 1j * k * z

        return e0 * self.orientation * np.exp(ikz)

    def current_density(self, xyz):
        r"""Current density for the harmonic planewave at a set of gridded locations.

        Parameters
        ----------
        xyz : (n, 3) numpy.ndarray
            Gridded xyz locations

        Returns
        -------
        (n_f, ..., 3) numpy.ndarray of complex
            Current density at all frequencies for the gridded
            locations provided.

        Examples
        --------
        Here, we define a harmonic planewave in the x-direction in a wholespace.

        >>> from geoana.em.fdem import HarmonicPlaneWave
        >>> import numpy as np
        >>> from geoana.utils import ndgrid
        >>> from mpl_toolkits.axes_grid1 import make_axes_locatable
        >>> import matplotlib.pyplot as plt

        Let us begin by defining the harmonic planewave in the x-direction.

        >>> frequency = 1
        >>> orientation = 'X'
        >>> sigma = 1.0
        >>> simulation = HarmonicPlaneWave(
        >>>     frequency=frequency, orientation=orientation, sigma=sigma
        >>> )

        Now we create a set of gridded locations and compute the current density.

        >>> x = np.linspace(-1, 1, 20)
        >>> z = np.linspace(-1000, 0, 20)
        >>> xyz = ndgrid(x, np.array([0]), z)
        >>> j_vec = simulation.current_density(xyz)
        >>> jx = j_vec[..., 0]
        >>> jy = j_vec[..., 1]
        >>> jz = j_vec[..., 2]

        Finally, we plot the real and imaginary parts of the x-oriented current density.

        >>> fig, axs = plt.subplots(2, 1, figsize=(14, 12))
        >>> titles = ['Real Part', 'Imaginary Part']
        >>> for ax, V, title in zip(axs.flatten(), [np.real(jx).reshape(20, 20), np.imag(jx).reshape(20, 20)], titles):
        >>>     im = ax.pcolor(x, z, V, shading='auto')
        >>>     divider = make_axes_locatable(ax)
        >>>     cax = divider.append_axes("right", size="5%", pad=0.05)
        >>>     cb = plt.colorbar(im, cax=cax)
        >>>     cb.set_label(label= 'Current Density ($A/m^2$)')
        >>>     ax.set_ylabel('Z coordinate ($m$)')
        >>>     ax.set_xlabel('X coordinate ($m$)')
        >>>     ax.set_title(title)
        >>> plt.tight_layout()
        >>> plt.show()
        """
        return self.sigma * self.electric_field(xyz)

    def magnetic_field(self, xyz):
        r"""Magnetic field for the harmonic planewave at a set of gridded locations.

        .. math::
            \nabla^2 \mathbf{H} + k^2 \mathbf{H} = 0

        where

        .. math::
            k = \sqrt{\omega^2 \mu \varepsilon - i \omega \mu \sigma}

        Parameters
        ----------
        xyz : (..., 3) numpy.ndarray
            Gridded xyz locations

        Returns
        -------
        (n_f, ..., 3) numpy.ndarray of complex
            Magnetic field at all frequencies for the gridded
            locations provided.

        Examples
        --------
        Here, we define a harmonic planewave in the x-direction in a wholespace.

        >>> from geoana.em.fdem import HarmonicPlaneWave
        >>> import numpy as np
        >>> from geoana.utils import ndgrid
        >>> from mpl_toolkits.axes_grid1 import make_axes_locatable
        >>> import matplotlib.pyplot as plt

        Let us begin by defining the harmonic planewave in the x-direction.

        >>> frequency = 1
        >>> orientation = 'X'
        >>> sigma = 1.0
        >>> simulation = HarmonicPlaneWave(
        >>>     frequency=frequency, orientation=orientation, sigma=sigma
        >>> )

        Now we create a set of gridded locations and compute the magnetic field.

        >>> x = np.linspace(-1, 1, 20)
        >>> z = np.linspace(-1000, 0, 20)
        >>> xyz = ndgrid(x, np.array([0]), z)
        >>> h_vec = simulation.magnetic_field(xyz)
        >>> hx = h_vec[..., 0]
        >>> hy = h_vec[..., 1]
        >>> hz = h_vec[..., 2]

        Finally, we plot the real and imaginary parts of the x-oriented magnetic field.

        >>> fig, axs = plt.subplots(2, 1, figsize=(14, 12))
        >>> titles = ['Real Part', 'Imaginary Part']
        >>> for ax, V, title in zip(axs.flatten(), [np.real(hy).reshape(20, 20), np.imag(hy).reshape(20, 20)], titles):
        >>>     im = ax.pcolor(x, z, V, shading='auto')
        >>>     divider = make_axes_locatable(ax)
        >>>     cax = divider.append_axes("right", size="5%", pad=0.05)
        >>>     cb = plt.colorbar(im, cax=cax)
        >>>     cb.set_label(label= 'Magnetic Field ($A/m$)')
        >>>     ax.set_ylabel('Z coordinate ($m$)')
        >>>     ax.set_xlabel('X coordinate ($m$)')
        >>>     ax.set_title(title)
        >>> plt.tight_layout()
        >>> plt.show()
        """
        return self.magnetic_flux_density(xyz) / self.mu

    def magnetic_flux_density(self, xyz):
        r"""Magnetic flux density for the harmonic planewave at a set of gridded locations.

        Parameters
        ----------
        xyz : (..., 3) numpy.ndarray
            Gridded xyz locations

        Returns
        -------
        (n_f, ..., 3) numpy.ndarray of complex
            Magnetic flux density at all frequencies for the gridded
            locations provided.

        Examples
        --------
        Here, we define a harmonic planewave in the x-direction in a wholespace.

        >>> from geoana.em.fdem import HarmonicPlaneWave
        >>> import numpy as np
        >>> from geoana.utils import ndgrid
        >>> from mpl_toolkits.axes_grid1 import make_axes_locatable
        >>> import matplotlib.pyplot as plt

        Let us begin by defining the harmonic planewave in the x-direction.

        >>> frequency = 1
        >>> orientation = 'X'
        >>> sigma = 1.0
        >>> simulation = HarmonicPlaneWave(
        >>>     frequency=frequency, orientation=orientation, sigma=sigma
        >>> )

        Now we create a set of gridded locations and compute the magnetic flux density.

        >>> x = np.linspace(-1, 1, 20)
        >>> z = np.linspace(-1000, 0, 20)
        >>> xyz = ndgrid(x, np.array([0]), z)
        >>> b_vec = simulation.magnetic_flux_density(xyz)
        >>> bx = b_vec[..., 0]
        >>> by = b_vec[..., 1]
        >>> bz = b_vec[..., 2]

        Finally, we plot the real and imaginary parts of the x-oriented magnetic flux density.

        >>> fig, axs = plt.subplots(2, 1, figsize=(14, 12))
        >>> titles = ['Real Part', 'Imaginary Part']
        >>> for ax, V, title in zip(axs.flatten(), [np.real(by).reshape(20, 20), np.imag(by).reshape(20, 20)], titles):
        >>>     im = ax.pcolor(x, z, V, shading='auto')
        >>>     divider = make_axes_locatable(ax)
        >>>     cax = divider.append_axes("right", size="5%", pad=0.05)
        >>>     cb = plt.colorbar(im, cax=cax)
        >>>     cb.set_label(label= 'Magnetic Flux Density (T)')
        >>>     ax.set_ylabel('Z coordinate ($m$)')
        >>>     ax.set_xlabel('X coordinate ($m$)')
        >>>     ax.set_title(title)
        >>> plt.tight_layout()
        >>> plt.show()
        """
        xyz = check_xyz_dim(xyz)
        z = xyz[..., [2]]

        k = append_ndim(self.wavenumber, xyz.ndim)
        omega = append_ndim(self.omega, xyz.ndim)
        e0 = self.amplitude
        kz = k * z
        ikz = 1j * kz

        # e = e0 * np.exp(ikz)[..., None]
        # Curl E = - i * omega * B
        # b = i / omega * d_z * e
        b = - e0 * k / omega * np.exp(ikz)

        # account for the orientation in the cross product
        # take cross product with the propagation direction
        b_dir = np.cross(self.orientation, [0, 0, -1])
        return b_dir * b
