# Here, we define a sphere with mass m and plot the gravitational
# potential as a function of distance.
#
import numpy as np
import matplotlib.pyplot as plt
from geoana.gravity import Sphere
from geoana.utils import ndgrid
#
# Define the sphere.
#
location = np.r_[0., 0., 0.]
rho = 1.0
radius = 1.0
simulation = Sphere(
    location=location, rho=rho, radius=radius
)
#
# Now we create a set of gridded locations, take the distances and compute the gravitational potential.
#
X, Y = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
Z = np.zeros_like(X) + 0.25
xyz = np.stack((X, Y, Z), axis=-1)
r = np.linalg.norm(xyz, axis=-1)
u = simulation.gravitational_potential(xyz)
#
# Finally, we plot the gravitational potential as a function of distance.
#
plt.plot(r, u)
plt.xlabel('Distance from sphere')
plt.ylabel('Gravitational potential')
plt.title('Gravitational Potential as a function of distance from sphere')
plt.show()
