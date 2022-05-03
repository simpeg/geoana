#!/usr/bin/env python
"""
geoana

Interactive geoscience (mostly) analytic functions.
"""

from distutils.core import setup
import sys

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
    version = '0.2.0',
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
        'empymod'
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
    or sys.argv[1] in ("--help-commands", "egg_info", "--version", "clean")
):
    # For these actions, NumPy is not required.
    #
    # They are required to succeed without Numpy, for example when
    # pip is used to install discretize when Numpy is not yet present in
    # the system.
    try:
        from setuptools import setup
    except ImportError:
        from distutils.core import setup
else:
    if (len(sys.argv) >= 2 and sys.argv[1] in ("bdist_wheel", "bdist_egg")) or (
        "develop" in sys.argv
    ):
        # bdist_wheel/bdist_egg needs setuptools
        import setuptools

    from numpy.distutils.core import setup

    # Add the configuration to the setup dict when building
    # after numpy is installed
    metadata["configuration"] = configuration


setup(**metadata)
