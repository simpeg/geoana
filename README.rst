| `getting_started`_ | `connecting`_ | `installing`_ | `license`_ | `documentation <http://geoana.readthedocs.io/en/latest/>`_ |

geoana
======

.. image:: https://readthedocs.org/projects/geoana/badge/?version=latest
    :target: https://geoana.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://img.shields.io/github/license/simpeg/geoana.svg
    :target: https://github.com/simpeg/geoana/blob/master/LICENSE
    :alt: MIT license

.. image:: https://travis-ci.org/simpeg/geoana.svg?branch=master
    :target: https://travis-ci.org/simpeg/geoana
    :alt: Travis status

.. image:: https://api.codacy.com/project/badge/Grade/2e32cd28f4424dc1800f1590a64c244f
    :target: https://www.codacy.com/app/lindseyheagy/geoana?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=simpeg/geoana&amp;utm_campaign=Badge_Grade
    :alt: codacy status


`geoana` is a collection of (mostly) analytic functions in geophysics. We take an object oriented
approach with the aim of having users be able to readily interact with the functions using `Jupyter <https://jupyter.org>`_


.. _getting_started:

Getting started
---------------

- If you do not already have python installed, we recommend downloading and installing it through `anaconda <https://www.anaconda.com/download/>`_
- :ref:`installing` geoana
- Browse the `gallery <http://geoana.readthedocs.io/en/latest/auto_examples/>`_ for ideas and example usage
- Read the `documentation <http://geoana.readthedocs.io/en/latest/>`_ for more information on the library and what it can do

.. - See the `contributor guide` and `code of conduct` if you are interested in helping develop or maintain geoana

.. _connecting:

Connecting with the community
-----------------------------

geoana is a part of the larger `SimPEG <https://simpeg.xyz>`_ ecosystem. There are several avenues for connecting:

- a mailing list for questions and general news items: https://groups.google.com/forum/#!forum/simpeg
- a newsletter where meeting notices and re-caps are posted: http://eepurl.com/bVUoOL
- a slack group for real-time chat with users and developers of SimPEG: http://slack.simpeg.xyz/

.. _installing:

Installing
----------

**geoana** is on conda-forge

.. code:: shell

    conda install -c conda-forge geoana

**geoana** is available on `pypi <https://pypi.org/project/geoana/>`_ and can be installed by opening a command window and running:

.. code::

    pip install geoana


To install from source, you can

.. code::

    git clone https://github.com/simpeg/geoana.git
    python setup.py install

.. _license:

License
-------

geoana is licensed under the `MIT license <https://github.com/simpeg/geoana/blob/master/LICENSE>`_ .
