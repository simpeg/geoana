# Reproducing part of Figure 4.8, scaled by magnetic suscpetibility, from Ward and
# Hohmann 1988.
#
import numpy as np
import matplotlib.pyplot as plt
from geoana.em.tdem import vertical_magnetic_flux_time_deriv_horizontal_loop
#
# Calculate the field at the time given
#
times = np.logspace(-7, -1)
dbz_dt = vertical_magnetic_flux_time_deriv_horizontal_loop(times, sigma=1E-2, radius=50)
#
# Match the vertical magnetic field plot
#
plt.loglog(times*1E3, -dbz_dt, '--')
plt.xlabel('time (ms)')
plt.ylabel(r'$\frac{\partial b_z}{ \partial t}$ (T/s)')
plt.show()
