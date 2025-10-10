import os
import pytest

try:
    from sphinx.application import Sphinx
except ImportError:
    Sphinx = None

@pytest.mark.skipif(Sphinx is None, reason="Sphinx not installed")
class TestDoc():

    @property
    def path_to_docs(self):
        dirname, file_name = os.path.split(os.path.abspath(__file__))
        return os.path.sep.join(dirname.split(os.path.sep)[:-1] + ["docs"])

    def test_html(self):
        src_dir = config_dir = self.path_to_docs
        output_dir = os.path.sep.join([src_dir, "_build", "html"])
        doctree_dir = os.path.sep.join([src_dir, "_build", "doctree"])
        app = Sphinx(src_dir, config_dir, output_dir, doctree_dir, buildername="html", warningiserror=False)
        app.build()

    def test_linkcheck(self):
        src_dir = config_dir = self.path_to_docs
        output_dir = os.path.sep.join([src_dir, "_build", "linkcheck"])
        doctree_dir = os.path.sep.join([src_dir, "_build", "doctree"])
        app = Sphinx(src_dir, config_dir, output_dir, doctree_dir, buildername="linkcheck", warningiserror=False)
        app.build()
