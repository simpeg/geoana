import unittest
from geoana.earthquake import oksar
import numpy as np


class TestOksar(unittest.TestCase):

    def test_los(self):

        dinar, dinar_fwd = oksar.example()

        dinar.ref = [741140, 4230327]
        dinar.ref_incidence = 23
        dinar.local_earth_radius = 6386232
        dinar.satellite_altitude = 788792
        dinar.satellite_azimuth = 192
        dinar.location_UTM_zone = 35

        utmLoc = np.array([706216.0606, 4269238.9999, 0])
        # refPoint = vmath.Vector3(dinar.ref.x, dinar.ref.y, 0)
        LOS = dinar.get_LOS_vector(utmLoc)

        # compare against fortran code.
        true = np.array([0.427051, -0.090772, 0.899660])
        np.testing.assert_allclose(true, LOS, rtol=1E-5)


if __name__ == '__main__':
    unittest.main()
