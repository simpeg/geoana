import numpy as np
from scipy.constants import mu_0
import properties
from scipy.special import iv, kv
from geoana.em.base import BaseMagneticDipole
from geoana.em.fdem.base import BaseFDEM


class MagneticDipoleHalfSpace(BaseMagneticDipole, BaseFDEM):
    """Harmonic magnetic dipole in a half space.

    Only valid for source and receivers at the surface. The surface is assumed
    to be at z=0.
    """

    frequency = properties.Array(
        "Source frequency (Hz)",
        shape=('*', ),
        dtype=float
    )

    @properties.validator("location")
    def _check_source_height(self, change):
        if change["value"][2] != 0.0:
            raise ValueError("Source must be at the surface of the earth (z=0)")

    def magnetic_field(self, xy, field="secondary"):
        """Magnetic field due to a magnetic dipole over a half space

        The analytic expression is only valid for a source and receiver at the
        surface of the earth. For arbitrary source and receiver locations above
        the earth, use the layered solution.

        Parameters
        ----------
        xy : numpy.ndarray
            receiver locations of shape (n_locations, 2)
        field : ("secondary", "total")
            Flag for the type of field to return.
        """
        sig = self.sigma_hat # (n_freq, )
        f = self.frequency
        w = 2*np.pi*f
        k = np.sqrt(-1j*w*mu_0*sig)[:, None] # This will get it to broadcast over locations
        dxy = xy[:, :2] - self.location[:2]
        r = np.linalg.norm(dxy, axis=-1)
        x = dxy[:, 0]
        y = dxy[:, 1]

        em_x = em_y = em_z = 0
        src_x, src_y, src_z = self.orientation
        # Z component of source
        alpha = 1j*k*r/2.
        IK1 = iv(1, alpha)*kv(1, alpha)
        IK2 = iv(2, alpha)*kv(2, alpha)
        if src_z != 0.0:
            em_z += src_z*2.0/(k**2*r**5)*(9-(9+9*1j*k*r-4*k**2*r**2-1j*k**3*r**3)*np.exp(-1j*k*r))
            Hr = (k**2/r)*(IK1 - IK2)
            angle = np.arctan2(y, x)
            em_x += src_z*np.cos(angle)*Hr
            em_y += src_z*np.sin(angle)*Hr

        if src_x != 0.0 or src_y != 0.0:
            # X component of source
            phi = 2/(k**2*r**4)*(3 + k**2*r**2 - (3 + 3j*k*r - k**2*r**2)*np.exp(-1j*k*r))
            dphi_dr = 2/(k**2*r**5)*(-2*k**2*r**2 - 12 + (-1j*k**3*r**3 - 5*k**2*r**2 + 12j*k*r + 12)*np.exp(-1j*k*r))
            if src_x != 0.0:
                em_x += src_x*(-1.0/r**3)*(y**2*phi + x**2*r*dphi_dr)
                em_y += src_x*(1.0/r**3)*x*y*(phi - r*dphi_dr)
                em_z -= src_x*(k**2*x/r**2)*(IK1 - IK2)

            # Y component of source
            if src_y != 0.0:
                em_x += src_y*(1.0/r**3)*x*y*(phi - r*dphi_dr)
                em_y += src_y*(-1.0/r**3)*(x**2*phi + y**2*r*dphi_dr)
                em_z -= src_y*(k**2*y/r**2)*(IK1 - IK2)

        if field == "secondary":
            # subtract out primary field from above
            mdotr = src_x*x + src_y*y# + m[2]*(z=0)

            em_x -= 3*x*mdotr/r**5 - src_x/r**3
            em_y -= 3*y*mdotr/r**5 - src_y/r**3
            em_z -= -src_z/r**3 # + 3*(z=0)*mdotr/r**5

        return self.moment/(4*np.pi)*np.stack((em_x, em_y, em_z), axis=-1)
