Conventions
===========

Coordinate systems
------------------

Cartesian
^^^^^^^^^

We use a right-handed coordinate system :math:`(x, y, z)` with z-positive up as shown in the Figure below.


Cylindrical
^^^^^^^^^^^

We again work with z-positive up and use :math:`\theta` to denote the azimuthal angle, thus the coordinate
system is defined as :math:`(r, \theta, z)`.


Spherical
^^^^^^^^^

We use :math:`r` for the radial direction, :math:`\theta` for the azimuthal direction, and :math:`\phi` for
the polar direction as shown in the figure below.


Fourier Transform
-----------------

For analysis and solutions in the frequency domain we use the :math:`e^{i \omega t}`
Fourier transform convention. Thus, we define our
Fourier Transform pair as

.. math ::
    F(\omega) = \int_{-\infty}^{\infty} f(t) e^{- i \omega t} dt \\

    f(t) = \frac{1}{2\pi}\int_{-\infty}^{\infty} F(\omega) e^{i \omega t} d \omega

where :math:`\omega` is angular frequency, :math:`t` is time, :math:`F(\omega)` is the
function defined in the frequency domain and :math:`f(t)` is the function defined in the time domain.
