import os
def configuration(parent_package="", top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration("kernels", parent_package, top_path)

    # Conditionally add subpackage if intending to build compiled components
    if os.environ.get('BUILD_GEOANA_EXT', "0") != "0":
        config.add_subpackage("_extensions")

    return config
