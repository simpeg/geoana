from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest

from geoana.em import static


class TestEM_Static(unittest.TestCase):

    def test_magnetic_dipole(self):

        edws = static.MagneticDipole_WholeSpace()
        print(edws.trait_names())


if __name__ == '__main__':
    unittest.main()
