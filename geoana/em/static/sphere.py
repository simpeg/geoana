import numpy as np
from scipy.constants import epsilon_0
from geoana.utils import check_xyz_dim


class ElectrostaticSphere:
    """Class for electrostatic solutions for a sphere in a wholespace.

    The ``ElectrostaticSphere`` class is used to analytically compute the electric
    potentials, fields, currents and charge densities for a sphere in a wholespace.
    For this class, we assume a homogeneous primary electric field along the
    :math:`\\hat{x}` direction.

    Parameters
    ----------
    radius : float
        radius of sphere (m).
    sigma_sphere : float
        conductivity of target sphere (S/m)
    sigma_background : float
        background conductivity (S/m)
    amplitude : float, optional
        amplitude of primary electric field along the :math:`\\hat{x}` direction (V/m).
        Default is 1.
    location : (3) array_like, optional
        Center of the sphere. Defaults is (0, 0, 0).
    """

    def __init__(
        self, radius, sigma_sphere, sigma_background, amplitude=1.0, location=np.r_[0.,0.,0.]
    ):

        self.radius = radius
        self.sigma_sphere = sigma_sphere
        self.sigma_background = sigma_background
        self.amplitude = amplitude
        self.location = location

    @property
    def sigma_sphere(self):
        """Electrical conductivity of the sphere in S/m

        Returns
        -------
        float
            Electrical conductivity of the sphere in S/m
        """
        return self._sigma_sphere

    @sigma_sphere.setter
    def sigma_sphere(self, item):
        item = float(item)
        if item <= 0.0:
            raise ValueError('Conductiviy must be positive')
        self._sigma_sphere = item

    @property
    def sigma_background(self):
        """Electrical conductivity of the background in S/m

        Returns
        -------
        float
            Electrical conductivity of the background in S/m
        """
        return self._sigma_background

    @sigma_background.setter
    def sigma_background(self, item):
        item = float(item)
        if item <= 0.0:
            raise ValueError('Conductiviy must be positive')
        self._sigma_background = item

    @property
    def radius(self):
        """Radius of the sphere in meters

        Returns
        -------
        float
            Radius of the sphere in meters
        """
        return self._radius

    @radius.setter
    def radius(self, item):
        item = float(item)
        if item < 0.0:
            raise ValueError('radius must be non-negative')
        self._radius = item

    @property
    def amplitude(self):
        """Amplitude of the primary current density along the x-direction.

        Returns
        -------
        float
            Amplitude of the primary current density along the x-direction in :math:`A/m^2`.
        """
        return self._amplitude

    @amplitude.setter
    def amplitude(self, item):
        self._amplitude = float(item)

    @property
    def location(self):
        """Center of the sphere

        Returns
        -------
        (3) numpy.ndarray of float
            Center of the sphere. Default = np.r_[0,0,0]
        """
        return self._location

    @location.setter
    def location(self, vec):

        try:
            vec = np.atleast_1d(vec).astype(float)
        except:
            raise TypeError(f"location must be array_like, got {type(vec)}")

        if len(vec) != 3:
            raise ValueError(
                f"location must be array_like with shape (3,), got {len(vec)}"
            )

        self._location = vec

    def potential(self, XYZ, field='all'):
        """Compute the electric potential.

        Parameters
        ----------
        XYZ : (3, ) tuple of np.ndarray or (..., 3) np.ndarray
            locations to evaluate at. If a tuple, all
            the numpy arrays must be the same shape.
        field : {'all', 'total', 'primary', 'secondary'}

        Returns
        -------
        Vt, Vp, Vs : (..., ) np.ndarray
            If field == "all"
        V : (..., ) np.ndarray
            If only requesting a single field.
        """
        XYZ = check_xyz_dim(XYZ)
        sig0 = self.sigma_background
        sig1 = self.sigma_sphere
        E0 = self.amplitude
        sig_cur = (sig1 - sig0) / (sig1 + 2 * sig0)
        r_vec = XYZ - self.location
        r = np.linalg.norm(r_vec, axis=-1)

        if field != 'total':
            Vp = -E0 * r_vec[..., 0]
            if field == 'primary':
                return Vp

        Vt = np.zeros_like(r)
        ind0 = r > self.radius
        # total potential outside the sphere
        Vt[ind0] = -E0*r_vec[ind0, 0]*(1.-sig_cur*self.radius**3./r[ind0]**3.)
        # inside the sphere
        Vt[~ind0] = -E0*r_vec[~ind0, 0]*3.*sig0/(sig1+2.*sig0)

        if field == 'total':
            return Vt
        # field was not primary or total
        Vs = Vt - Vp
        if field == 'secondary':
            return Vs
        return Vt, Vp, Vs

    def electric_field(self, XYZ, field='all'):
        """Electric field for a sphere in a uniform wholespace

        Parameters
        ----------
        XYZ : (3, ) tuple of np.ndarray or (..., 3) np.ndarray
            locations to evaluate at. If a tuple, all
            the numpy arrays must be the same shape.
        field : {'all', 'total', 'primary', 'secondary'}

        Returns
        -------
        Et, Ep, Es : (..., 3) np.ndarray
            If field == "all"
        E : (..., 3) np.ndarray
            If only requesting a single field.
        """
        XYZ = check_xyz_dim(XYZ)
        sig0 = self.sigma_background
        sig1 = self.sigma_sphere
        E0 = self.amplitude
        sig_cur = (sig1 - sig0) / (sig1 + 2 * sig0)
        r_vec = XYZ - self.location
        r = np.linalg.norm(r_vec, axis=-1)

        if field != 'total':
            Ep = np.zeros_like(r_vec)
            Ep[..., 0] = E0
            if field == 'primary':
                return Ep

        Et = np.zeros_like(r_vec)
        ind0 = r > self.radius
        # total field outside the sphere
        Et[ind0, 0] = E0 + E0*self.radius**3./(r[ind0]**5.)*sig_cur*(2.*r_vec[ind0, 0]**2.-r_vec[ind0, 1]**2.-r_vec[ind0, 2]**2.)
        Et[ind0, 1] = E0*self.radius**3./(r[ind0]**5.)*3.*r_vec[ind0, 0]*r_vec[ind0, 1]*sig_cur
        Et[ind0, 2] = E0*self.radius**3./(r[ind0]**5.)*3.*r_vec[ind0, 0]*r_vec[ind0, 2]*sig_cur
        # inside the sphere
        Et[~ind0, 0] = 3.*sig0/(sig1+2.*sig0)*E0

        if field == 'total':
            return Et
        # field was not primary or total
        Es = Et - Ep
        if field == 'secondary':
            return Es
        return Et, Ep, Es

    def current_density(self, XYZ, field='all'):
        """Current density for a sphere in a uniform wholespace

        Parameters
        ----------
        XYZ : (3, ) tuple of np.ndarray or (..., 3) np.ndarray
            locations to evaluate at. If a tuple, all
            the numpy arrays must be the same shape.
        field : {'all', 'total', 'primary', 'secondary'}

        Returns
        -------
        Jt, Jp, Js : (..., 3) np.ndarray
            If field == "all"
        J : (..., 3) np.ndarray
            If only requesting a single field.
        """
        XYZ = check_xyz_dim(XYZ)

        Et, Ep, Es = self.electric_field(XYZ, field='all')
        if field != 'total':
            Jp = self.sigma_background * Ep
            if field == 'primary':
                return Jp
        r_vec = XYZ - self.location
        r = np.linalg.norm(r_vec, axis=-1)

        sigma = np.full(r.shape, self.sigma_background)
        sigma[r <= self.radius] = self.sigma_sphere

        Jt = sigma[..., None] * Et
        if field == 'total':
            return Jt

        Js = Jt - Jp
        if field == 'secondary':
            return Js
        return Jt, Jp, Js

    def charge_density(self, XYZ, dr=None):
        """charge density on the surface of a sphere in a uniform wholespace

        Parameters
        ----------
        XYZ : (3, ) tuple of np.ndarray or (..., 3) np.ndarray
            locations to evaluate at. If a tuple, all
            the numpy arrays must be the same shape.
        dr : float, optional
            Buffer around the edge of the sphere to calculate
            current density. Defaults to 5 % of the sphere radius

        Returns
        -------
        rho: (..., ) np.ndarray
        """
        XYZ = check_xyz_dim(XYZ)

        sig0 = self.sigma_background
        sig1 = self.sigma_sphere
        sig_cur = (sig1 - sig0) / (sig1 + 2 * sig0)
        Ep = self.electric_field(XYZ, field='primary')

        r_vec = XYZ - self.location
        r = np.linalg.norm(r_vec, axis=-1)

        if dr is None:
            dr = 0.05 * self.radius

        ind = (r < self.radius + 0.5*dr) & (r > self.radius - 0.5*dr)

        rho = np.zeros_like(r)
        rho[ind] = epsilon_0*3.*Ep[ind, 0]*sig_cur*x[ind]/(np.sqrt(x[ind]**2.+y[ind]**2.))

        return rho
