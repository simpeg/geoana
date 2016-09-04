from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest

from geoana.em import fdem


class TestFDEM(unittest.TestCase):

    def test_vector(self):
        edws = fdem.ElectricDipole_WholeSpace()

if __name__ == '__main__':
    unittest.main()
