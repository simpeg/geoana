#!/usr/bin/env python
"""
geoana

Interactive geoscience (mostly) analytic functions.
"""

import sys
from setuptools import setup

CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Education',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Scientific/Engineering :: Physics',
    'Operating System :: Microsoft :: Windows',
    'Operating System :: POSIX',
    'Operating System :: Unix',
    'Operating System :: MacOS',
    'Natural Language :: English',
]

def configuration(parent_package="", top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path)
    config.set_options(
        ignore_setup_xxx_py=True,
        assume_default_configuration=True,
        delegate_options_to_subpackages=True,
        quiet=True,
    )

    config.add_subpackage("geoana")

    return config

with open('README.rst') as f:
    LONG_DESCRIPTION = ''.join(f.readlines())

metadata = dict(
    name = 'geoana',
    version = '0.4.1',
    python_requires=">=3.6",
    setup_requires=[
        "numpy>=1.8",
        "cython>=0.2",
    ],
    install_requires = [
        'numpy>=1.8',
        'scipy>=0.13',
        'matplotlib',
        'utm',
        'empymod>=2.0'
    ],
    author = 'SimPEG developers',
    author_email = 'lindseyheagy@gmail.com',
    description = 'Analytic expressions for geophysical responses',
    long_description = LONG_DESCRIPTION,
    keywords = 'geophysics, electromagnetics',
    url = 'https://www.simpeg.xyz',
    download_url = 'https://github.com/simpeg/geoana',
    classifiers=CLASSIFIERS,
    platforms = ['Windows', 'Linux', 'Solaris', 'Mac OS-X', 'Unix'],
    license='MIT License'
)

if len(sys.argv) >= 2 and (
    "--help" in sys.argv[1:]
    or sys.argv[1] in ("--help-commands", "egg_info", "install_egg_info", "--version", "clean")
):
    # For these actions, NumPy is not required.
    #
    # They are required to succeed without Numpy, for example when
    # pip is used to install geoana when Numpy is not yet present in
    # the system.

    setup_requires = metadata['setup_requires']
    install_requires = metadata['install_requires']
    install_requires = setup_requires + install_requires[1:]
    metadata['install_requires'] = install_requires
else:
    from numpy.distutils.core import setup

    # Add the configuration to the setup dict when building
    # after numpy is installed
    metadata["configuration"] = configuration


setup(**metadata)
