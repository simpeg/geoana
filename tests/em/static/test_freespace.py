import numpy as np
import numpy.testing as npt

import pytest
from scipy.special import roots_legendre

from geoana.em.static import MagneticPrism, MagneticDipoleWholeSpace

class TestMagneticAccuracy():
    x, y, z = np.mgrid[-100:100:20j, -100:100:20j, -100:100:20j]
    xyz = np.stack((x, y, z), axis=-1).reshape((-1, 3))
    dx = 0.1
    prism = MagneticPrism(dx * np.r_[-1, -1, -1], dx * np.r_[1, 1, 1], magnetization=[-1, 2, -0.5])
    m_mag = np.linalg.norm(prism.magnetization)
    m_unit = prism.magnetization/m_mag
    dipole = MagneticDipoleWholeSpace(
        location=[0, 0, 0], moment=m_mag, orientation=m_unit
    )

    quad_points, quad_weights = roots_legendre(5)
    quad_points = (prism.max_location - prism.min_location)[:, None] * (quad_points + 1) / 2 + prism.min_location[:,
                                                                                               None]
    quad_points = np.stack(np.meshgrid(*quad_points, indexing='ij'), axis=-1)
    quad_wx, quad_wy, quad_wz = quad_weights * (prism.max_location - prism.min_location)[:, None] / 2
    quad_wx = quad_wx[:, None, None]
    quad_wy = quad_wy[None, :, None]
    quad_wz = quad_wz[None, None, :]

    quad_xyzs = xyz - quad_points[..., None, :]

    @pytest.mark.parametrize(
        'method,rtol',
        [
            ('magnetic_field', 1E-7),
            ('magnetic_flux_density', 1E-7),
         ]
    )
    def test_accuracy(self, method, rtol):
        test_prism = getattr(self.prism, method)(self.xyz)

        wx = append_ndim(self.quad_wx, test_prism.ndim)
        wy = append_ndim(self.quad_wy, test_prism.ndim)
        wz = append_ndim(self.quad_wz, test_prism.ndim)

        test_quad = getattr(self.dipole, method)(self.quad_xyzs)
        test_quad *= wx
        test_quad *= wy
        test_quad *= wz
        test_quad = np.sum(test_quad, axis=(0, 1, 2))

        atol = rtol * (test_prism.max() - test_prism.min())
        npt.assert_allclose(test_quad, test_prism, atol=atol)