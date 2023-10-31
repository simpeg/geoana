from . import utils
from . import earthquake
from . import em

__author__ = 'SimPEG developers'
__license__ = 'MIT'
__copyright__ = 'Copyright 2023 SimPEG developers'

from importlib.metadata import version, PackageNotFoundError

# Version
try:
    # - Released versions just tags:       0.8.0
    # - GitHub commits add .dev#+hash:     0.8.1.dev4+g2785721
    # - Uncommitted changes add timestamp: 0.8.1.dev4+g2785721.d20191022
    __version__ = version("geoana")
except PackageNotFoundError:
    # If it was not installed, then we don't know the version. We could throw a
    # warning here, but this case *should* be rare. geoana should be
    # installed properly!
    from datetime import datetime

    __version__ = "unknown-" + datetime.today().strftime("%Y%m%d")


try:
    import geoana.kernels._extensions.rTE
    import geoana.kernels._extensions.potential_field_prism
    compiled = True
except ImportError:
    compiled = False
def show_config():
    info_dict = {
        'version': __version__,
        'compiled': compiled,
    }
    print(info_dict)
    return info_dict