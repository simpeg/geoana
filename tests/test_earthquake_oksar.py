from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
from geoana.earthquake import oksar
import vectormath as vmath


class TestOksar(unittest.TestCase):

    def test_los(self):

        dinar, dinar_fwd = oksar.example()

        dinar.ref = [741140, 4230327]
        dinar.ref_incidence = 23
        dinar.local_earth_radius = 6386232
        dinar.satellite_altitude = 788792
        dinar.satellite_azimuth = 192
        dinar.location_UTM_zone = 35

        utmLoc = vmath.Vector3([706216.0606], [4269238.9999], [0])
        # refPoint = vmath.Vector3(dinar.ref.x, dinar.ref.y, 0)

        LOS = dinar.get_LOS_vector(utmLoc)

        # compare against fortran code.
        true = vmath.Vector3([0.427051, -0.090772, 0.899660])
        assert (LOS - true).length < 1e-5


if __name__ == '__main__':
    unittest.main()
