import numpy as np
from geoana.em.fdem.base import BaseFDEM
from geoana.spatial import repeat_scalar
from geoana.em.base import BaseElectricDipole, BaseMagneticDipole


class ElectricDipoleWholeSpace(BaseFDEM, BaseElectricDipole):
    r"""Class for a harmonic electric dipole in a wholespace.

    Harmonic electric dipole in a whole space. The source is
    (c.f. Ward and Hohmann, 1988 page 173). The source current
    density for a dipole located at :math:`\mathbf{r}_s` with orientation
    :math:`\mathbf{\hat{u}}`

    .. math::
        \mathbf{J}(\\mathbf{r}) = I ds \delta(\mathbf{r}
        - \mathbf{r}_s)\mathbf{\hat{u}}

    """

    def __init__(self, frequency, **kwargs):

        BaseFDEM.__init__(self, frequency, **kwargs)
        BaseElectricDipole.__init__(self, **kwargs)


    def vector_potential(self, xyz):
        r"""Vector potential for the harmonic current dipole at a set of gridded locations.

        For an electric current dipole oriented in the :math:`\hat{u}` direction with
        dipole moment :math:`I ds`, the magnetic vector potential at frequency :math:`f`
        at vector distance :math:`\mathbf{r}` from the dipole is given by:

        .. math::
            \mathbf{a}(\mathbf{r}) = \frac{I ds}{4 \pi r} e^{-ikr} \mathbf{\hat{u}}

        where

        .. math::
            k = \sqrt{\omega**2 \mu \varepsilon - i \omega \mu \sigma}

        Parameters
        ----------
        xyz : (n, 3) numpy.ndarray
            Gridded xyz locations

        Returns
        -------
        (n, 3) numpy.array of complex
            Magnetic vector potential at the gridded location provided.

        """
        r = self.distance(xyz)
        a = (
            (self.current * self.length) / (4*np.pi*r) *
            np.exp(-1j*self.wavenumber*r)
        )
        a = np.kron(np.ones(1, 3), np.atleast_2d(a).T)
        return self.dot_orientation(a)

    def electric_field(self, xyz):
        r"""Electric field for the harmonic current dipole at a set of gridded locations.

        For an electric current dipole oriented in the :math:`\hat{u}` direction with
        dipole moment :math:`I ds` and harmonic frequency :math:`f`, this method computes the
        electric field at the set of gridded xyz locations provided.

        The analytic solution is adapted from Ward and Hohmann (1988). For a harmonic electric
        current dipole oriented in the :math:`\hat{x}` direction, the solution at vector distance
        :math:`\mathbf{r}` from the current dipole is:

        .. math::
            \mathbf{E}(\mathbf{r}) = \frac{I ds}{4\pi (\sigma + i \omega \varepsilon) r^3} e^{-ikr} \Bigg [ \Bigg ( \frac{x^2}{r^2}\hat{x} + \frac{xy}{r^2}\hat{y} + \frac{xz}{r^2}\hat{z} \Bigg ) ... \\
            \big ( -k^2r^2 + 3ikr + 3 \big ) \big ( k^2 r^2 - ikr - 1 \big ) \hat{x} \Bigg ]
        
        where

        .. math::
            k = \sqrt{\omega**2 \mu \varepsilon - i \omega \mu \sigma}

        Parameters
        ----------
        xyz : (n, 3) numpy.ndarray
            Gridded xyz locations

        Returns
        -------
        (n, 3) numpy.array of complex
            Electric field at the gridded locations provided.

        """
        dxyz = self.vector_distance(xyz)
        r = repeat_scalar(self.distance(xyz))
        kr = self.wavenumber * r
        ikr = 1j * kr

        front_term = (
            (self.current * self.length) / (4 * np.pi * self.sigma * r**3) *
            np.exp(-ikr)
        )
        symmetric_term = (
            repeat_scalar(self.dot_orientation(dxyz)) * dxyz *
            (-kr**2 + 3*ikr + 3) / r**2
        )
        oriented_term = (
            (kr**2 - ikr - 1) *
            np.kron(self.orientation, np.ones((dxyz.shape[0], 1)))
        )
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
            \mathbf{J}(\mathbf{r}) = \frac{\sigma I ds}{4\pi (\sigma + i \omega \varepsilon) r^3} e^{-ikr} \Bigg [ \Bigg ( \frac{x^2}{r^2}\hat{x} + \frac{xy}{r^2}\hat{y} + \frac{xz}{r^2}\hat{z} \Bigg ) ... \\
            \big ( -k^2r^2 + 3ikr + 3 \big ) \big ( k^2 r^2 - ikr - 1 \big ) \hat{x} \Bigg ]
        
        where

        .. math::
            k = \sqrt{\omega**2 \mu \varepsilon - i \omega \mu \sigma}

        Parameters
        ----------
        xyz : (n, 3) numpy.ndarray
            Gridded xyz locations

        Returns
        -------
        (n, 3) numpy.array of complex
            Current density at the gridded locations provided.

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
            k = \sqrt{\omega**2 \mu \varepsilon - i \omega \mu \sigma}

        Parameters
        ----------
        xyz : (n, 3) numpy.ndarray
            Gridded xyz locations

        Returns
        -------
        (n, 3) numpy.array of complex
            Magnetic field at the gridded locations provided.

        """
        dxyz = self.vector_distance(xyz)
        r = repeat_scalar(self.distance(xyz))
        kr = self.wavenumber * r
        ikr = 1j*kr

        front_term = (
            self.current * self.length / (4 * np.pi * r**2) * (ikr + 1) *
            np.exp(-ikr)
        )
        return -front_term * self.cross_orientation(dxyz) / r

    def magnetic_flux_density(self, xyz):
        r"""Magnetic flux density produced by the harmonic current dipole at a set of gridded locations.

        For an electric current dipole oriented in the :math:`\hat{u}` direction with
        dipole moment :math:`I ds` and harmonic frequency :math:`f`, this method computes the
        magnetic flux density at the set of gridded xyz locations provided.

        The analytic solution is adapted from Ward and Hohmann (1988). For a harmonic electric
        current dipole oriented in the :math:`\hat{x}` direction, the solution at vector distance
        :math:`\mathbf{r}` from the current dipole is:

        .. math::
            \mathbf{B}(\mathbf{r}) = \frac{\mu I ds}{4\pi r^2} (ikr + 1) e^{-ikr} \big ( - \frac{z}{r}\hat{y} + \frac{y}{r}\hat{z} \big )
        
        where

        .. math::
            k = \sqrt{\omega**2 \mu \varepsilon - i \omega \mu \sigma}

        Parameters
        ----------
        xyz : (n, 3) numpy.ndarray
            Gridded xyz locations

        Returns
        -------
        (n, 3) numpy.array of complex
            Magnetic flux density at the gridded locations provided.

        """
        return self.mu * self.magnetic_field(xyz)


class MagneticDipoleWholeSpace(BaseMagneticDipole, BaseFDEM):
    """
    Harmonic magnetic dipole in a whole space.
    """

    def __init__(self, frequency, **kwargs):

        BaseFDEM.__init__(self, frequency, **kwargs)
        BaseMagneticDipole.__init__(self, **kwargs)

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
            k = \sqrt{\omega**2 \mu \varepsilon - i \omega \mu \sigma}

        Parameters
        ----------
        xyz : (n, 3) numpy.ndarray
            Gridded xyz locations

        Returns
        -------
        (n, 3) numpy.array of complex
            Magnetic vector potential at the gridded location provided.

        """
        r = self.distance(xyz)
        f = (
            (1j * self.omega * self.mu * self.moment) / (4 * np.pi * r) *
            np.exp(-1j * self.wavenumber * r)
        )
        f = np.kron(np.ones(1, 3), np.atleast_2d(f).T)
        return self.dot_orientation(f)

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
            k = \sqrt{\omega**2 \mu \varepsilon - i \omega \mu \sigma}

        Parameters
        ----------
        xyz : (n, 3) numpy.ndarray
            Gridded xyz locations

        Returns
        -------
        (n, 3) numpy.array of complex
            Electric field at the gridded locations provided.

        """
        dxyz = self.vector_distance(xyz)
        r = repeat_scalar(self.distance(xyz))
        kr = self.wavenumber*r
        ikr = 1j * kr

        front_term = (
            (1j * self.omega * self.mu * self.moment) / (4. * np.pi * r**2) *
            (ikr + 1) * np.exp(-ikr)
        )
        return front_term * self.cross_orientation(dxyz) / r

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
            k = \sqrt{\omega**2 \mu \varepsilon - i \omega \mu \sigma}

        Parameters
        ----------
        xyz : (n, 3) numpy.ndarray
            Gridded xyz locations

        Returns
        -------
        (n, 3) numpy.array of complex
            Current density at the gridded locations provided.

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
            k = \sqrt{\omega**2 \mu \varepsilon - i \omega \mu \sigma}

        Parameters
        ----------
        xyz : (n, 3) numpy.ndarray
            Gridded xyz locations

        Returns
        -------
        (n, 3) numpy.array of complex
            Magnetic field at the gridded locations provided.

        """
        dxyz = self.vector_distance(xyz)
        r = repeat_scalar(self.distance(xyz))
        kr = self.wavenumber*r
        ikr = 1j*kr

        front_term = self.moment / (4. * np.pi * r**3) * np.exp(-ikr)
        symmetric_term = (
            repeat_scalar(self.dot_orientation(dxyz)) * dxyz *
            (-kr**2 + 3*ikr + 3) / r**2
        )
        oriented_term = (
            (kr**2 - ikr - 1) *
            np.kron(self.orientation, np.ones((dxyz.shape[0], 1)))
        )

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
            k = \sqrt{\omega**2 \mu \varepsilon - i \omega \mu \sigma}

        Parameters
        ----------
        xyz : (n, 3) numpy.ndarray
            Gridded xyz locations

        Returns
        -------
        (n, 3) numpy.array of complex
            Magnetic flux density at the gridded locations provided.

        """
        return self.mu * self.magnetic_field(xyz)
