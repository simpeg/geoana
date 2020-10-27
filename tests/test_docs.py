from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import subprocess
import unittest


class TestDoc(unittest.TestCase):

    @property
    def path_to_docs(self):
        dirname, filename = os.path.split(os.path.abspath(__file__))
        return os.path.sep.join(dirname.split(os.path.sep)[:-1] + ['docs'])

    def test_html(self):
        wd = os.getcwd()
        os.chdir(self.path_to_docs)

        response = subprocess.run(["make", "html"])
        self.assertTrue(response.returncode == 0)
        os.chdir(wd)

    def test_linkcheck(self):
        wd = os.getcwd()
        os.chdir(self.path_to_docs)

        response = subprocess.run(["make", "linkcheck"])
        print(response.returncode)
        self.assertTrue(response.returncode == 0)
        os.chdir(wd)


if __name__ == '__main__':
    unittest.main()
