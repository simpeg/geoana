def configuration(parent_package="", top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration("geoana", parent_package, top_path)

    config.add_subpackage("earthquake")
    config.add_subpackage("em")
    config.add_subpackage("kernels")

    return config
