from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest

from geoana.em import static


class TestEM_Static(unittest.TestCase):

    def test_magnetic_dipole(self):

        mdws = static.MagneticDipole_WholeSpace()

        # with warnings.catch_warnings(record=True) as w:
        #     warnings.simplefilter("always")
        #     mdws.sigma = 2
        #     assert len(w) == 1

        # print(mdws.traits.trait_names())


if __name__ == '__main__':
    unittest.main()
