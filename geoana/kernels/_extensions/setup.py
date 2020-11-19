import os
import os.path

base_path = os.path.abspath(os.path.dirname(__file__))

def configuration(parent_package="", top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs

    config = Configuration("_extensions", parent_package, top_path)

    ext = "rTE"
    try:
        from Cython.Build import cythonize

        cythonize(os.path.join(base_path, "rTE.pyx"))
    except ImportError:
        pass

    config.add_extension(
        ext, sources=["rTE.cpp", "_rTE.cpp"], include_dirs=[get_numpy_include_dirs()]
    )

    return config
