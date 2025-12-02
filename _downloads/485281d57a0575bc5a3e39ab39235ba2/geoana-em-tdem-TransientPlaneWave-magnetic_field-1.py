# Here, we define a transient planewave in the x-direction in a wholespace.
#
from geoana.em.tdem import TransientPlaneWave
import numpy as np
from geoana.utils import ndgrid
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
#
# Let us begin by defining the transient planewave in the x-direction.
#
time = 1.0
orientation = 'X'
sigma = 1.0
simulation = TransientPlaneWave(
    time=time, orientation=orientation, sigma=sigma
)
#
# Now we create a set of gridded locations and compute the electric field.
#
x = np.linspace(-1, 1, 20)
z = np.linspace(-1000, 0, 20)
xyz = ndgrid(x, np.array([0]), z)
h_vec = simulation.magnetic_field(xyz)
hy = h_vec[..., 1]
#
# Finally, we plot the x-oriented magnetic field.
#
plt.pcolor(x, z, hy.reshape(20, 20), shading='auto')
cb = plt.colorbar()
cb.set_label(label= 'Magnetic Field ($A/m$)')
plt.ylabel('Z coordinate ($m$)')
plt.xlabel('X coordinate ($m$)')
plt.title('Magnetic Field of a Transient Planewave in the x-direction in a Wholespace')
plt.show()
