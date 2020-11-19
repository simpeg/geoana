import properties
import numpy as np
from geoana.em.base import BaseMagneticDipole
from geoana.em.fdem.base import BaseFDEM, sigma_hat
from scipy.constants import mu_0, epsilon_0
from empymod.utils import check_hankel
from empymod.transform import get_dlf_points
from geoana.kernels.tranverse_electric_reflections import rTE_forward


class MagneticDipoleLayeredHalfSpace(BaseMagneticDipole, BaseFDEM):

    thickness = properties.Array(
        "Layer thicknesses (m) starting from the top-most layer. The bottom layer is assumed to be infinite.",
        shape=('*', ),
        dtype=float
    )

    sigma = properties.Array(
        "Electrical conductivity (S/m), defined starting from the top most layer",
        shape=('*', ),
        dtype=complex,
        coerce=True
    )

    frequency = properties.Array(
        "Source frequency (Hz)",
        shape=('*', ),
        dtype=float
    )

    mu = properties.Array(
        "Magnetic permeability (H/m), defined starting from the top most layer",
        shape=('*', ),
        dtype=complex,
        default=np.array([mu_0], dtype=np.complex128)
    )

    epsilon = properties.Array(
        "Permitivity value (F/m), defined starting from the top most layer",
        shape=('*', ),
        dtype=float,
        default=np.array([epsilon_0], dtype=np.float64)
    )

    def _get_valid_properties(self):
        thick = self.thickness
        n_layer = len(thick)+1
        sigma = self.sigma
        epsilon = self.epsilon
        mu = self.mu
        if n_layer != 1:
            sigma = self.sigma
            if len(sigma) == 1:
                sigma = np.ones(n_layer)*sigma
            epsilon = self.epsilon
            if len(epsilon) == 1:
                epsilon = np.ones(n_layer)*epsilon
            mu = self.mu
            if len(mu) == 1:
                mu = np.ones(n_layer)*mu
        return thick, sigma, epsilon, mu

    @property
    def sigma_hat(self):
        _, sigma, epsilon, _ = self._get_valid_properties()
        return sigma_hat(
            self.frequency[:, None], sigma, epsilon,
            quasistatic=self.quasistatic
        ).T

    @property
    def wavenumber(self):
        raise NotImplementedError()

    @property
    def skin_depth(self):
        raise NotImplementedError()

    def magnetic_field(self, xyz, field="secondary"):
        """
        Magnetic field due to a magnetic dipole in a layered halfspace at a specific height z

        Parameters
        ----------
        xyz : numpy.ndarray
            receiver locations of shape (n_locations, 3).
            The z component cannot be below the surface (z=0.0).
        field : ("secondary", "total")
            Flag for the type of field to return.
        """

        if np.any(xyz[:, 2] < 0.0):
            raise ValueError("Cannot compute fields below the surface")
        h = self.location[0]
        dxyz = xyz - self.location
        offsets = np.linalg.norm(dxyz[:, :-1], axis=-1)

        # Comput transform operations
        # -1 gives lagged convolution in dlf
        ht, htarg = check_hankel('dlf', {'dlf': 'key_101_2009', 'pts_per_dec': 0}, 1)
        fhtfilt = htarg['dlf']
        pts_per_dec = htarg['pts_per_dec']

        f = self.frequency
        n_frequency = len(f)

        lambd, int_points = get_dlf_points(fhtfilt, offsets, pts_per_dec)

        thick = self.thickness
        n_layer = len(thick) + 1

        thick, sigma, epsilon, mu = self._get_valid_properties()
        sigh = sigma_hat(
            self.frequency[:, None], sigma, epsilon,
            quasistatic=self.quasistatic
        ).T  # this gets sigh with proper shape (n_layer x n_freq) and fortran ordering.
        mu = np.tile(mu, (n_frequency, 1)).T  # shape(n_layer x n_freq)

        rTE = rTE_forward(f, lambd.reshape(-1), sigh, mu, thick)
        rTE = rTE.reshape((n_frequency, *lambd.shape))

        # secondary is height of receiver plus height of source
        rTE *= np.exp(-lambd*(xyz[:, -1] + h)[:, None])
        # works for variable xyz because each point has it's own lambdas

        src_x, src_y, src_z = self.orientation
        C0x = C0y = C0z = 0.0
        C1x = C1y = C1z = 0.0
        if src_x != 0.0:
            C0x += src_x*(dxyz[:, 0]**2/offsets**2)[:, None]*lambd**2
            C1x += src_x*(1/offsets - 2*dxyz[:, 0]**2/offsets**3)[:, None]*lambd
            C0y += src_x*(dxyz[:, 0]*dxyz[:, 1]/offsets**2)[:, None]*lambd**2
            C1y -= src_x*(2*dxyz[:, 0]*dxyz[:, 1]/offsets**3)[:, None]*lambd
            # C0z += 0.0
            C1z -= (src_x*dxyz[:, 0]/offsets)[:, None]*lambd**2

        if src_y != 0.0:
            C0x += src_y*(dxyz[:, 0]*dxyz[:, 1]/offsets**2)[:, None]*lambd**2
            C1x -= src_y*(2*dxyz[:, 0]*dxyz[:, 1]/offsets**3)[:, None]*lambd
            C0y += src_y*(dxyz[:, 1]**2/offsets**2)[:, None]*lambd**2
            C1y += src_y*(1/offsets - 2*dxyz[:, 1]**2/offsets**3)[:, None]*lambd
            # C0z += 0.0
            C1z -= (src_y*dxyz[:, 1]/offsets)[:, None]*lambd**2

        if src_z != 0.0:
            # C0x += 0.0
            C1x += (src_z*dxyz[:, 0]/offsets)[:, None]*lambd**2
            # C0y += 0.0
            C1y += (src_z*dxyz[:, 1]/offsets)[:, None]*lambd**2
            C0z += src_z*lambd**2
            # C1z += 0.0

        # Do the hankel transform on each component
        em_x = ((C0x*rTE)@fhtfilt.j0 + (C1x*rTE)@fhtfilt.j1)/offsets
        em_y = ((C0y*rTE)@fhtfilt.j0 + (C1y*rTE)@fhtfilt.j1)/offsets
        em_z = ((C0z*rTE)@fhtfilt.j0 + (C1z*rTE)@fhtfilt.j1)/offsets

        if field == "total":
            # add in the primary field
            r = np.linalg.norm(dxyz, axis=-1)
            mdotr = src_x*dxyz[:, 0] + src_y*dxyz[:, 1] + src_z*dxyz[:, 2]

            em_x += 3*dxyz[:, 0]*mdotr/r**5 - src_x/r**3
            em_y += 3*dxyz[:, 1]*mdotr/r**5 - src_y/r**3
            em_z += 3*dxyz[:, 2]*mdotr/r**5 - src_z/r**3

        return self.moment/(4*np.pi)*np.stack((em_x, em_y, em_z), axis=-1)
