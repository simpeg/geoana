# Here, we define a harmonic planewave in the x-direction in a wholespace.
#
from geoana.em.fdem import HarmonicPlaneWave
import numpy as np
from geoana.utils import ndgrid
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
#
# Let us begin by defining the harmonic planewave in the x-direction.
#
frequency = 1
orientation = 'X'
sigma = 1.0
simulation = HarmonicPlaneWave(
    frequency=frequency, orientation=orientation, sigma=sigma
)
#
# Now we create a set of gridded locations and compute the magnetic field.
#
x = np.linspace(-1, 1, 20)
z = np.linspace(-1000, 0, 20)
xyz = ndgrid(x, np.array([0]), z)
h_vec = simulation.magnetic_field(xyz)
hx = h_vec[..., 0]
hy = h_vec[..., 1]
hz = h_vec[..., 2]
#
# Finally, we plot the real and imaginary parts of the x-oriented magnetic field.
#
fig, axs = plt.subplots(2, 1, figsize=(14, 12))
titles = ['Real Part', 'Imaginary Part']
for ax, V, title in zip(axs.flatten(), [np.real(hy).reshape(20, 20), np.imag(hy).reshape(20, 20)], titles):
    im = ax.pcolor(x, z, V, shading='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(im, cax=cax)
    cb.set_label(label= 'Magnetic Field ($A/m$)')
    ax.set_ylabel('Z coordinate ($m$)')
    ax.set_xlabel('X coordinate ($m$)')
    ax.set_title(title)
plt.tight_layout()
plt.show()
