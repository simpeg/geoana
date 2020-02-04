"""
oksar3

Program to calcuate forward models of interferograms, strain tensor, etc.
from Okada subroutine.

Heritage:
    - originally fringes.c written by Barry Parsons
    - updated to oksar                                 tjw
    - oksar_strain: added strain tensor calculation    tjw
    - oksar3:       added new line of sight calculator tjw feb 2003
    - Modified into Python by RowanCockett, 3point Science Aug 2014


"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import properties
import vectormath as vmath
import utm
import matplotlib.pyplot as plt


class EarthquakeInterferogram(properties.HasProperties):
    title = properties.String(
        'name of the earthquake',
        required=True
    )

    description = properties.String(
        'description of the event',
        required=False
    )

    location = properties.Vector2(
        'interferogram location (bottom N, left E)',
        required=True
    )

    location_UTM_zone = properties.Integer(
        'UTM zone',
        required=True
    )

    shape = properties.Array(
        'number of pixels in the interferogram',
        shape=(2,),
        dtype=int,
        required=True
    )

    pixel_size = properties.Array(
        'Size of each pixel (northing, easting)',
        shape=(2,),
        dtype=float,
        required=True
    )

    data = properties.Array(
        'Processed interferogram data (unwrapped)',
        dtype=float,
        required=True
    )

    ref = properties.Vector2(
        'interferogram reference',
        required=True
    )

    ref_incidence = properties.Float(
        'Incidence angle',
        required=True
    )

    scaling = properties.Float(
        'Scaling of the interferogram',
        default=1.0
    )

    satellite_name = properties.String('Name of the satelite.')

    satellite_fringe_interval = properties.Float(
        'Fringe interval',
        default=0.028333
    )

    satellite_azimuth = properties.Float(
        'satellite_azimuth',
        required=True
    )

    satellite_altitude = properties.Float(
        'satellite_altitude',
        required=True
    )

    local_rigidity = properties.Float(
        'Local rigidity',
        default=3e10
    )

    local_earth_radius = properties.Float(
        'Earth radius',
        default=6371000.
    )

    date1 = properties.DateTime(
        'date1',
        required=True
    )

    date2 = properties.DateTime(
        'date2',
        required=True
    )

    processed_by = properties.String(
        'processed_by',
        required=True
    )

    processed_date = properties.DateTime(
        'processed_date',
        required=True
    )

    copyright = properties.String(
        'copyright',
        required=True
    )

    data_source = properties.String(
        'data_source',
        required=True
    )

    event_date = properties.DateTime('Date of the earthquake')
    event_gcmt_id = properties.String('GCMT ID')
    event_name = properties.String('Earthquake name')
    event_country = properties.String('Earthquake country')

    def _get_plot_data(self):

        vectorNx = (
            np.r_[
                0,
                np.cumsum(
                    (self.pixel_size[0],) * self.shape[0]
                )
            ] + self.location[0]
        )
        vectorNy = (
            np.r_[
                0,
                np.cumsum(
                    (self.pixel_size[1],) * self.shape[1]
                )
            ] + self.location[1]
        ) - self.pixel_size[1] * self.shape[1]

        data = self.data.copy()
        data = np.flipud(data.reshape(self.shape, order='F').T)
        data[data == 0] = np.nan
        data *= self.scaling

        return vectorNx, vectorNy, data

    def plot_interferogram(self, wrap=True, ax=None):

        self.assert_valid

        if ax is None:
            plt.figure()
            ax = plt.subplot(111)

        vectorNx, vectorNy, data = self._get_plot_data()

        if wrap:
            cmap = plt.cm.hsv
            data = data % self.satellite_fringe_interval
            vmin, vmax = 0.0, self.satellite_fringe_interval
        else:
            cmap = plt.cm.jet
            vmin = np.nanmin(data)
            vmax = np.nanmax(data)

        out = ax.pcolormesh(
            vectorNx,
            vectorNy,
            np.ma.masked_where(np.isnan(data), data),
            vmin=vmin,
            vmax=vmax,
            cmap=cmap
        )

        ax.set_title(self.title)
        ax.axis('equal')
        ax.set_xlabel('Easting, m (UTM Zone {})'.format(
            self.location_UTM_zone
        ))
        ax.set_ylabel('Northing, m')

        cb = plt.colorbar(out, ax=ax)
        cb.set_label('Displacement, m')

        return out

    def plot_mask(self, ax=None, opacity=0.2):

        if ax is None:
            plt.figure()
            ax = plt.subplot(111)

        vectorNx, vectorNy, data = self._get_plot_data()

        from matplotlib import colors
        cmap = colors.ListedColormap([(1, 1, 1, opacity)])

        out = ax.pcolormesh(
            vectorNx,
            vectorNy,
            np.ma.masked_where(~np.isnan(data), data),
            cmap=cmap
        )

        ax.set_title(self.title)
        ax.axis('equal')
        ax.set_xlabel('Easting, m (UTM Zone {})'.format(
            self.location_UTM_zone
        ))
        ax.set_ylabel('Northing, m')

        return out

    def get_LOS_vector(self, locations):
        """
        calculate beta - the angle at earth center between reference point
        and satellite nadir
        """
        if not isinstance(locations, list):
            locations = [locations]

        utmZone = self.location_UTM_zone
        refPoint = vmath.Vector3(self.ref.x, self.ref.y, 0)
        satAltitude = self.satellite_altitude
        satAzimuth = self.satellite_azimuth
        satIncidence = self.ref_incidence
        earthRadius = self.local_earth_radius

        DEG2RAD = np.pi / 180.
        alpha = satIncidence * DEG2RAD
        beta = (earthRadius / (satAltitude + earthRadius)) * np.sin(alpha)
        beta = alpha - np.arcsin(beta)
        beta = beta / DEG2RAD

        # calculate angular separation of (x,y) from satellite track passing
        # through (origx, origy) with azimuth satAzimuth

        # Long lat **NOT** lat long
        origy, origx = utm.to_latlon(
            refPoint.x, refPoint.y, np.abs(utmZone), northern=utmZone > 0
        )

        xy = np.array([
            utm.to_latlon(u[0], u[1], np.abs(utmZone), northern=utmZone > 0)
            for u in locations
        ])
        y = xy[:, 0]
        x = xy[:, 1]

        angdist = self._ang_to_gc(x, y, origx, origy, satAzimuth)

        # calculate beta2, the angle at earth center between roaming point and
        # satellite nadir track, assuming right-looking satellite

        beta2 = beta - angdist
        beta2 = beta2 * DEG2RAD

        # calculate alpha2, the new incidence angle

        alpha2 = np.sin(beta2) / (
            np.cos(beta2) - (earthRadius / (earthRadius + satAltitude))
        )
        alpha2 = np.arctan(alpha2)
        alpha2 = alpha2 / DEG2RAD

        # calculate pointing vector

        satIncidence = 90 - alpha2
        satAzimuth = 360 - satAzimuth

        los_x = -np.cos(satAzimuth * DEG2RAD) * np.cos(satIncidence * DEG2RAD)
        los_y = -np.sin(satAzimuth * DEG2RAD) * np.cos(satIncidence * DEG2RAD)
        los_z = np.sin(satIncidence * DEG2RAD)

        return vmath.Vector3Array([los_x, los_y, los_z])

    @staticmethod
    def _ang_to_gc(x, y, origx, origy, satAzimuth):
        """
        Calculate angular distance to great circle passing through
        given point
        """

        Ngc = np.zeros(3)
        cartxy = np.zeros((len(x), 3))
        satAzimuth = np.deg2rad(satAzimuth)
        origx = np.deg2rad(origx)
        origy = np.deg2rad(origy)

        x = np.deg2rad(x)
        y = np.deg2rad(y)

        # 1. calc geocentric norm vec to great circle, Ngc = Rz*Ry*Rx*[0;1;0]
        #    where Rz = rotation of origx about geocentric z-axis
        #    where Ry = rotation of origy about geocentric y-axis
        #    where Rx = rotation of satAzimuth about geocentric x-axis
        #    and [0;1;0] is geocentric norm vec to N-S Great Circle through 0 0

        Ngc[0] = (
                    (np.sin(satAzimuth) * np.sin(origy) * np.cos(origx)) -
                    (np.cos(satAzimuth) * np.sin(origx))
                 )
        Ngc[1] = (
                    (np.sin(satAzimuth) * np.sin(origy) * np.sin(origx)) +
                    (np.cos(satAzimuth) * np.cos(origx))
                 )
        Ngc[2] = -np.sin(satAzimuth) * np.cos(origy)

        # 2. calculate unit vector geocentric coordinates for lon/lat
        #    position (x,y)

        cartxy[:, 0] = np.cos(x) * np.cos(y)
        cartxy[:, 1] = np.sin(x) * np.cos(y)
        cartxy[:, 2] = np.sin(y)

        # 3. Dot product between Ngc and cartxy gives angle 90 degrees
        #    bigger than what we want

        angdist = (
                        Ngc[0]*cartxy[:, 0] +
                        Ngc[1]*cartxy[:, 1] +
                        Ngc[2]*cartxy[:, 2]
                  )

        angdist = np.rad2deg(np.arccos(angdist)) - 90

        return angdist


class Oksar(properties.HasProperties):

    beta = properties.Float('beta', default=3E10)
    mu = properties.Float('mu', default=3E10)

    strike = properties.Float('Strike', min=0, max=360)
    dip = properties.Float('Dip', default=45, min=0, max=90)
    rake = properties.Float('Rake', default=90, min=-180, max=180)
    slip = properties.Float('Slip', default=0.5, min=0)
    length = properties.Float('Fault length', default=10000., min=0)
    center = properties.Vector2('Center of the fault plane.')
    depth_top = properties.Float('Top of fault', min=0)
    depth_bottom = properties.Float('Bottom of fault', default=10000, min=0)

    O = properties.Vector2(
        'Origin of the simulation domain', required=True
    )

    U = properties.Vector2(
        'U direction of the simulation domain', required=True
    )

    V = properties.Vector2(
        'V direction of the simulation domain', required=True
    )

    shape = properties.Array(
        'number of pixels in the simulation',
        shape=(2,),
        default=(300, 300),
        dtype=int,
        # required=True
    )

    @property
    def simulation_grid(self):

        self.assert_valid

        vec, shape = vmath.ouv2vec(
            vmath.Vector3(self.O[0], self.O[1], 0),
            vmath.Vector3(self.U[0], self.U[1], 0),
            vmath.Vector3(self.V[0], self.V[1], 0),
            self.shape
        )
        return vec

    @property
    def displacement_vector(self):

        self.assert_valid

        vec = self.simulation_grid
        x, y = vec.x, vec.y

        DEG2RAD = 0.017453292519943
        alpha = (self.beta + self.mu) / (self.beta + 2.0 * self.mu)

        #  Here we could loop over models

        flt_x = self.center[0]
        flt_y = self.center[1]
        strike = self.strike
        dip = self.dip
        rake = self.rake
        slip = self.slip
        length = self.length
        hmin = self.depth_top
        hmax = self.depth_bottom

        rrake = (rake+90.0)*DEG2RAD
        sindip = np.sin(dip*DEG2RAD)
        w = (hmax-hmin)/sindip
        ud = slip*np.cos(rrake)
        us = -slip*np.sin(rrake)
        halflen = length/2.0
        al2 = halflen
        al1 = -al2
        aw1 = hmin/sindip
        aw2 = hmax/sindip

        if(hmin < 0.0):
            raise Exception('ERROR: Fault top above ground surface')

        if(hmin == 0.0):
            hmin = 0.00001

        sstrike = (strike+90.0)*DEG2RAD

        ct = np.cos(sstrike)
        st = np.sin(sstrike)

        X = ct * (-flt_x + x) - st * (-flt_y + y)
        Y = ct * (-flt_y + y) + st * (-flt_x + x)

        u = self._dc3d3(alpha, X, Y, -dip, al1, al2, aw1, aw2, us, ud)

        UX = ct*u.x + st*u.y
        UY = -st*u.x + ct*u.y
        UZ = u.z

        return vmath.Vector3(UX, UY, UZ)

    def _dc3d3(self, alpha, X, Y, dip, al1, al2, aw1, aw2, disl1, disl2):
        F0 = 0.0
        F1 = 1.0
        F2 = 2.0
        PI2 = 6.283185307179586
        EPS = 1.0E-6

        u = vmath.Vector3(F0, F0, F0)
        dub = vmath.Vector3(F0, F0, F0)

        #  %%dccon0 subroutine
        #  Calculates medium and fault dip constants
        c0_alp3 = (F1 - alpha) / alpha
        #  PI2/360
        pl8 = 0.017453292519943
        c0_sd = np.sin(dip*pl8)
        c0_cd = np.cos(dip*pl8)

        if(np.abs(c0_cd) < EPS):
            c0_cd = F0
            if(c0_sd > F0):
                c0_sd = F1

            if(c0_sd < F0):
                c0_sd = -F1

        c0_cdcd = c0_cd * c0_cd
        c0_sdcd = c0_sd * c0_cd

        #  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        p = Y * c0_cd
        q = Y * c0_sd

        jxi = ((X - al1) * (X - al2)) <= F0  # BOOLEAN
        jet = ((p - aw1) * (p - aw2)) <= F0  # BOOLEAN

        for k in [1., 2.]:
            et = 0.0
            if(k == 1):
                et = p-aw1
            else:
                et = p-aw2

            for j in [1., 2.]:
                xi = 0.0
                if(j == 1):
                    xi = X-al1
                else:
                    xi = X-al2

                # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                # %%dccon2 subroutine
                # % calculates station geometry constants for finite source

                dc_max = np.max(np.abs(np.c_[xi, et, q]))

                # dc_max = max(np.abs(xi),max(np.abs(et),np.abs(q)))

                xi[(np.abs(xi/dc_max) < EPS) | (np.abs(xi) < EPS)] = F0

                et[(np.abs(et/dc_max) < EPS) | (np.abs(et) < EPS)] = F0

                q[(np.abs(q/dc_max) < EPS) | (np.abs(q) < EPS)] = F0

                dc_xi = xi
                dc_et = et
                dc_q = q
                c2_r = np.sqrt(dc_xi*dc_xi + dc_et*dc_et + dc_q*dc_q)

                if np.any(c2_r == F0):
                    raise Exception('singularity error ???')

                c2_y = dc_et * c0_cd + dc_q * c0_sd
                c2_d = dc_et * c0_sd - dc_q * c0_cd
                c2_tt = np.arctan(dc_xi * dc_et / (dc_q * c2_r))
                c2_tt[dc_q == F0] = F0

                rxi = c2_r + dc_xi
                c2_x11 = F1/(c2_r*rxi)
                c2_x11[(dc_xi < F0) & (dc_q == F0) & (dc_et == F0)] = F0

                ret = c2_r + dc_et
                if np.any(ret < 1e-14):
                    raise Exception('dccon2 b %f %f %f %f' % (
                        ret, c2_r, dc_et, dc_q, dc_xi
                    ))

                c2_ale = np.log(ret)
                c2_y11 = F1/(c2_r*ret)

                ind = (dc_et < F0) & (dc_q == F0) & (dc_xi == F0)

                # if((c2_r-dc_et) < 1e-14):
                #     raise Exception('dccon2 a %f %f %f %f %f' % (
                #         c2_3-dc_et, c2_r, dc_et, dc_q, dc_xi)
                #     )

                c2_ale[ind] = -np.log(c2_r[ind]-dc_et[ind])
                c2_y11[ind] = F0

                # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                if np.any(
                    (
                        (q == F0) &
                        (
                            ((jxi) & (et == F0)) |
                            ((jet) & (xi == F0))
                        )
                    ) | (c2_r == F0)
                ):
                    raise Exception('singular problems: 2')

                # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                # ub subroutine
                # part B of displacement and strain at depth due to buried
                # faults in semi-infinite medium

                rd = c2_r + c2_d
                if np.any(rd < 1e-14):
                    raise Exception('ub %f %f %f %f %f %f' % (
                        rd, c2_r, c2_d, xi, et, q
                    ))

                ai3 = 0.0
                ai4 = 0.0
                if(c0_cd != F0):
                    # xx replaces x in original subroutine
                    xx = np.sqrt(xi*xi+q*q)
                    ai4 = F1/c0_cdcd * (xi/rd*c0_sdcd + F2*np.arctan(
                        (
                            et*(xx+q*c0_cd) +
                            xx*(c2_r+xx)*c0_sd
                        ) / (xi*(c2_r+xx)*c0_cd)
                    ))
                    ai4[xi == F0] = F0

                    ai3 = (c2_y*c0_cd/rd - c2_ale + c0_sd*np.log(rd)) / c0_cdcd
                else:
                    rd2 = rd*rd
                    ai3 = (et/rd + c2_y*q/rd2 - c2_ale) / F2
                    ai4 = xi*c2_y/rd2/F2

                ai1 = -xi/rd*c0_cd - ai4*c0_sd
                ai2 = np.log(rd) + ai3*c0_sd
                qx = q*c2_x11
                qy = q*c2_y11

                # strike-slip contribution
                if(disl1 != 0.0):
                    du2x = - xi*qy - c2_tt - c0_alp3 * ai1 * c0_sd
                    du2y = - q/c2_r + c0_alp3*c2_y/rd*c0_sd
                    du2z = q*qy - c0_alp3*ai2*c0_sd
                    du2 = vmath.Vector3(du2x, du2y, du2z)
                    dub = du2 * (disl1 / PI2)
                else:
                    dub = vmath.Vector3()

                # dip-slip contribution
                if(disl2 != F0):
                    du2x = - q/c2_r + c0_alp3 * ai3 * c0_sdcd
                    du2y = - et*qx - c2_tt - c0_alp3 * xi / rd * c0_sdcd
                    du2z = q*qx + c0_alp3 * ai4 * c0_sdcd
                    du2 = vmath.Vector3(du2x, du2y, du2z)
                    dub = dub + (du2 * (disl2 / PI2))

                # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                dux = dub.x
                duy = dub.y*c0_cd - dub.z*c0_sd
                duz = dub.y*c0_sd + dub.z*c0_cd
                du = vmath.Vector3(dux, duy, duz)
                if((j+k) != 3):
                    u = + du + u
                else:
                    u = - du + u

        return u

    def plot_displacement(self, eq=None, ax=None, wrap=True, mask_opacity=0.2):

        if eq is not None:
            assert isinstance(eq, EarthquakeInterferogram)

        if ax is None:
            plt.figure()
            ax = plt.subplot(111)

        vectorNx = (
            np.r_[
                0,
                np.cumsum(
                    (self.U[0]/self.shape[0],) * self.shape[0]
                )
            ] + self.O[0]
        )
        vectorNy = (
            np.r_[
                0,
                np.cumsum(
                    (self.V[1]/self.shape[1],) * self.shape[1]
                )
            ] + self.O[1]
        )

        DIR = self.displacement_vector
        grid = self.simulation_grid
        if eq is None:
            LOS = vmath.Vector3(0, 0, 1)
        else:
            LOS = eq.get_LOS_vector(grid)
        data = DIR.dot(LOS)
        data = np.flipud(data.reshape(self.shape, order='F').T)
        # data[data == 0] = np.nan
        # data *= self.scaling

        if wrap and eq is not None:
            cmap = plt.cm.hsv
            data = data % eq.satellite_fringe_interval
            vmin, vmax = 0.0, eq.satellite_fringe_interval
        else:
            cmap = plt.cm.jet
            vmin = np.nanmin(data)
            vmax = np.nanmax(data)

        out = ax.pcolormesh(
            vectorNx,
            vectorNy,
            np.ma.masked_where(np.isnan(data), data),
            vmin=vmin,
            vmax=vmax,
            cmap=cmap
        ),

        ax.axis('equal')
        ax.set_xlabel('Easting, m')
        ax.set_ylabel('Northing, m')

        cb = plt.colorbar(out[0], ax=ax)
        cb.set_label('Displacement, m')

        if eq is not None:
            mask = eq.plot_mask(ax=ax, opacity=mask_opacity)
            out = out[0], mask

        return out


def example():

    import requests
    dinar_file = requests.get(
        'https://storage.googleapis.com/simpeg/geoana/dinar.r4'
    )
    data = np.fromstring(dinar_file.content, np.float32)

    dinar = EarthquakeInterferogram(
        # uid='dinar',
        title='Dinar, Turkey',
        description=(
            'On October 1, 1995, a strong earthquake ruptured a section of '
            'the Dinar-Civril fault in SW Turkey. Around 30% of the buildings '
            'in the nearby town of Dinar were destroyed. 92 inhabitants were '
            'killed and over 200 injured.'
        ),
        event_country='Turkey',
        event_date='1995-09-30T18:00:00Z',
        event_gcmt_id='100195B',
        event_name='Dinar',
        copyright='ESA',
        data_source='ESA',
        date1='1995-08-12T18:00:00Z',
        date2='1995-12-31T17:00:00Z',
        processed_by='GarethFunning',
        processed_date='2003-01-20T17:00:00Z',
        ref_incidence=23,
        ref=[741140., 4230327.],
        scaling=0.0045040848895,
        local_earth_radius=6386232,
        local_rigidity=30000000000,
        location=[706216.0606, 4269238.9999],
        shape=(1024, 1024),
        location_UTM_zone=35,
        pixel_size=[80., 80.],
        satellite_altitude=788792,
        satellite_azimuth=192,
        satellite_fringe_interval=0.028333333,
        satellite_name='ERS',
        data=data
    )

    dinar_fwd = Oksar(
        beta=3E10,
        mu=3E10,
        strike=329.6,
        dip=50,
        rake=90,
        slip=0.5,
        length=11578.907244622129,
        center=[773728.2977967655, 4223586.816611591],
        depth_top=0,
        depth_bottom=15000,
        O=[706216.0606, 4187318.9999],
        U=[81920, 0],
        V=[0, 81920],
        shape=(300, 200)
    )

    return dinar, dinar_fwd
