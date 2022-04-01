import os
import subprocess
import unittest
import platform

use_shell = platform.system() == "Windows"


class TestDoc(unittest.TestCase):

    @property
    def path_to_docs(self):
        dirname, file_name = os.path.split(os.path.abspath(__file__))
        return os.path.sep.join(dirname.split(os.path.sep)[:-1] + ["docs"])

    def test_html(self):
        os.chdir(self.path_to_docs)

        response = subprocess.run(["make", "html"], shell=use_shell)
        self.assertTrue(response.returncode == 0)

    def test_linkcheck(self):
        os.chdir(self.path_to_docs)

        response = subprocess.run(["make", "linkcheck"], shell=use_shell)
        self.assertTrue(response.returncode == 0)

if __name__ == '__main__':
    unittest.main()
