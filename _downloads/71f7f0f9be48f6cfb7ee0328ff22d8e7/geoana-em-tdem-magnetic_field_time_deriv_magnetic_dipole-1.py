# Reproducing the time derivate parts of Figure 4.4 and 4.5 from Ward and Hohmann 1988
#
import numpy as np
import matplotlib.pyplot as plt
from geoana.em.tdem import magnetic_field_time_deriv_magnetic_dipole
#
# Calculate the field at the time given, 100 m along the x-axis,
#
times = np.logspace(-6, 0, 200)
xy = np.array([[100, 0, 0]])
dh_dt = magnetic_field_time_deriv_magnetic_dipole(times, xy, sigma=1E-2)
#
# Match the vertical magnetic field plot
#
plt.loglog(times*1E3, dh_dt[:,0, 2], c='C0', label=r'$\frac{\partial h_z}{\partial t}$')
plt.loglog(times*1E3, -dh_dt[:,0, 2], '--', c='C0')
plt.loglog(times*1E3, dh_dt[:,0, 0], c='C1', label=r'$\frac{\partial h_x}{\partial t}$')
plt.loglog(times*1E3, -dh_dt[:,0, 0], '--', c='C1')
plt.xlabel('time (ms)')
plt.ylabel(r'$\frac{\partial h}{\partial t}$ (A/(m s))')
plt.legend()
plt.show()
