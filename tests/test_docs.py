from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import subprocess
import unittest
import platform


class TestDoc(unittest.TestCase):

    @property
    def path_to_docs(self):
        dirname, file_name = os.path.split(os.path.abspath(__file__))
        return dirname.split(os.path.sep)[:-1] + ["docs"]

    def test_html(self):
        wd = os.getcwd()
        os.chdir(os.path.sep.join(self.path_to_docs))

        if platform.system() != "Windows":
            response = subprocess.run(["make", "html"])
            self.assertTrue(response.returncode == 0)
        else:
            response = subprocess.call(["make", "html"], shell=True)
            self.assertTrue(response == 0)

        os.chdir(wd)

    def test_linkcheck(self):
        wd = os.getcwd()
        os.chdir(os.path.sep.join(self.path_to_docs))

        if platform.system() != "Windows":
            response = subprocess.run(["make", "linkcheck"])
            self.assertTrue(response.returncode == 0)
        else:
            response = subprocess.call(["make", "linkcheck"], shell=True)
            self.assertTrue(response == 0)

        os.chdir(wd)


if __name__ == '__main__':
    unittest.main()
