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

import numpy as np
import utm
import matplotlib.pyplot as plt
from datetime import datetime

def _date_time_from_json(value):
    if len(value) == 10:
        return datetime.strptime(value.replace('-', '/'),'%Y/%m/%d')
    return datetime.strptime(value, '%Y-%m-%dT%H:%M:%SZ')


class EarthquakeInterferogram:
    """Interferogram class"""

    def __init__(
        self,
        data,
        title,
        location,
        location_UTM_zone,
        shape,
        pixel_size,
        ref,
        ref_incidence,
        satellite_azimuth,
        satellite_altitude,
        processed_by,
        scaling=1.0,
        satellite_fringe_interval=0.028333,
        local_earth_radius=6371000.,
        local_rigidity=3e10,
        **kwargs
    ):

        self.data = data
        self.title = title
        self.location = location
        self.location_UTM_zone = location_UTM_zone
        self.shape = shape
        self.pixel_size = pixel_size
        self.ref = ref
        self.ref_incidence = ref_incidence
        self.satellite_azimuth = satellite_azimuth
        self.satellite_altitude = satellite_altitude
        self.processed_by = processed_by
        self.scaling = scaling
        self.satellite_fringe_interval = satellite_fringe_interval
        self.local_earth_radius = local_earth_radius
        self.local_rigidity = local_rigidity

        kwargs_list = [
            'description', 'event_country', 'event_name', 'copyright', 'data_source',
            'satellite_name', 'event_gcmt_id', 'date1', 'date2', 'processed_date', 'event_date'
        ]

        for k in kwargs_list:
            if k in kwargs.keys():
                setattr(self, k, kwargs.pop(k))

        super().__init__(**kwargs)

    @property
    def data(self):
        """Processed interferogram data (unwrapped).

        Returns
        -------
        numpy.ndarray of float
            Processed interferogram data (unwrapped)

        """
        return self._data

    @data.setter
    def data(self, var):
        try:
            var = np.asarray(var, dtype=float)
        except:
            raise TypeError("Location must be a array_like of float")

        self._data = var

    @property
    def title(self):
        """A title.

        Returns
        -------
        str
            A title.

        """
        return self._title

    @title.setter
    def title(self, var):
        try:
            var = str(var)
        except:
            raise TypeError("Title is not a string or data type that can be converted to string")
        self._title = var

    @property
    def location(self):
        """Interferogram location (bottom N, left E).

        Returns
        -------
        (2,) numpy.ndarray
            Interferogram location (bottom N, left E)

        """
        return self._location

    @location.setter
    def location(self, var):
        try:
            var = np.asarray(var, dtype=float)
        except:
            raise TypeError(
                "Location must be array_like of float"
            )
        var = np.squeeze(var)
        if var.shape != (2, ):
            raise ValueError(
                f"Location must be of shape (2,) not {var.shape}"
            )

        self._location = var

    @property
    def location_UTM_zone(self):
        """UTM zone for the interferogram.

        Returns
        -------
        int
            UTM zone for the interferogram

        """
        return self._location_UTM_zone

    @location_UTM_zone.setter
    def location_UTM_zone(self, var):
        try:
            var = int(var)
        except:
            raise TypeError("Location_UTM_zone must be integer")

        self._location_UTM_zone = var

    @property
    def shape(self):
        """Shape of the interferogram defined by number of pixels in
        the North and Easting directions, respectively; i.e. (n_pix_N, n_pix_E).

        Returns
        -------
        (2,) tuple of int
            Interferogram shape, (number in Northing, number in Easting).

        """
        return self._shape

    @shape.setter
    def shape(self, var):
        try:
            var = tuple(int(v) for v in var)
        except:
            raise TypeError("shape must be a array_like of int")

        if len(var) != 2:
            raise ValueError(f"shape must be length 2, not{len(var)}")

        self._shape = var

    @property
    def pixel_size(self):
        """The Northing and Easting dimensions of each pixel.

        Returns
        -------
        (2,) numpy.ndarray of float
            The Northing and Easting dimensions of each pixel.

        """
        return self._pixel_size

    @pixel_size.setter
    def pixel_size(self, var):
        try:
            var = np.asarray(var, dtype=float)
        except:
            raise TypeError("pixel_size must be array_like of float")
        var = np.squeeze(var)
        if var.shape != (2,):
            raise TypeError(f"pixel_size must have shape (2, ) not {var.shape}")

        self._pixel_size = var

    @property
    def ref(self):
        """Interferogram reference location (bottom N, left E).

        Returns
        -------
        (2,) numpy.ndarray
            Interferogram reference location (bottom N, left E)

        """
        return self._ref

    @ref.setter
    def ref(self, var):
        try:
            var = np.asarray(var, dtype=float)
        except:
            raise TypeError(
                "ref must be a array_like of float"
            )
        var = np.squeeze(var)
        if var.shape != (2, ):
            raise ValueError(f"ref must have shape (2, ) not {var.shape}")

        self._ref = var

    @property
    def ref_incidence(self):
        """Incidence angle.

        Returns
        -------
        float
            Incidence angle

        """
        return self._ref_incidence

    @ref_incidence.setter
    def ref_incidence(self, var):
        try:
            var = float(var)
        except:
            raise TypeError("ref_incidence must be a float")

        self._ref_incidence = var

    @property
    def scaling(self):
        """Interferogram scaling factor.

        Returns
        -------
        float
            Interferogram scaling factor. Default = 1.0

        """
        return self._scaling

    @scaling.setter
    def scaling(self, var):
        try:
            var = float(var)
        except:
            raise TypeError("scaling must be a float")

        self._scaling = var

    @property
    def satellite_fringe_interval(self):
        """Satellite fringe interval.

        Returns
        -------
        float
            Satellite fringe interval

        """
        return self._satellite_fringe_interval

    @satellite_fringe_interval.setter
    def satellite_fringe_interval(self, var):
        try:
            var = float(var)
        except:
            raise TypeError("satellite_fringe_interval must be a float")

        self._satellite_fringe_interval = var

    @property
    def satellite_azimuth(self):
        """Satellite azimuth.

        Returns
        -------
        float
            Satellite azimuth

        """
        return self._satellite_azimuth

    @satellite_azimuth.setter
    def satellite_azimuth(self, var):
        try:
            var = float(var)
        except:
            raise TypeError("satellite_azimuth must be a float")

        self._satellite_azimuth = var

    @property
    def satellite_altitude(self):
        """Satellite altitude.

        Returns
        -------
        float
            Satellite altitude

        """
        return self._satellite_altitude

    @satellite_altitude.setter
    def satellite_altitude(self, var):
        try:
            var = float(var)
        except:
            raise TypeError("satellite_altitude must be a float")

        self._satellite_altitude = var

    @property
    def local_earth_radius(self):
        """Local Earth radius.

        Returns
        -------
        float
            Local Earth radius. Default = 6371000.

        """
        return self._local_earth_radius

    @local_earth_radius.setter
    def local_earth_radius(self, var):
        try:
            var = float(var)
        except:
            raise TypeError("local_earth_radius must be a float")

        self._local_earth_radius = var

    @property
    def local_rigidity(self):
        """Local rigidity.

        Returns
        -------
        float
            Local rigidity. Default = 3e10

        """
        return self._local_rigidity

    @local_rigidity.setter
    def local_rigidity(self, var):
        try:
            var = float(var)
        except:
            raise TypeError("local_rigidity must be a float")

        self._rigidity = var

    @property
    def processed_by(self):
        """A string stating who processed the data.

        Returns
        -------
        str
            A string stating who processed the data.

        """
        return self._processed_by

    @processed_by.setter
    def processed_by(self, var):
        try:
            var = str(var)
        except:
            raise TypeError("processed_by not a string or data type that can be converted to string")

        self._processed_by = var

    @property
    def description(self):
        """The description of the event.

        Returns
        -------
        str
            The description event. Default = 'My Earthquake'

        """
        return self._description

    @description.setter
    def description(self, var):
        try:
            var = str(var)
        except:
            raise TypeError("description is not a string or data type that can be converted to string")
        self._description = var

    @property
    def satellite_name(self):
        """A string stating the name of the satellite.

        Returns
        -------
        str
            A string stating who the name of the satellite.

        """
        return self._satellite_name

    @satellite_name.setter
    def satellite_name(self, var):
        try:
            var = str(var)
        except:
            raise TypeError("satellite_name not a string or data type that can be converted to string")

        self._satellite_name = var

    @property
    def copyright(self):
        """A string providing any copyright information.

        Returns
        -------
        str
            A string providing any copyright information.

        """
        return self._copyright

    @copyright.setter
    def copyright(self, var):
        try:
            var = str(var)
        except:
            raise TypeError("copyright not a string or data type that can be converted to string")

        self._copyright = var

    @property
    def data_source(self):
        """The source of the data.

        Returns
        -------
        str
            The source of the data.

        """
        return self._data_source

    @data_source.setter
    def data_source(self, var):
        try:
            var = str(var)
        except:
            raise TypeError("data_source not a string or data type that can be converted to string")

        self._data_source = var

    @property
    def event_gcmt_id(self):
        """The GCMT ID for the event.

        Returns
        -------
        str
            The GCMT ID for the event.

        """
        return self._event_gcmt_id

    @event_gcmt_id.setter
    def event_gcmt_id(self, var):
        try:
            var = str(var)
        except:
            raise TypeError("event_gcmt_id not a string or data type that can be converted to string")

        self._event_gcmt_id = var

    @property
    def event_name(self):
        """The name of the event.

        Returns
        -------
        str
            The name of the event.

        """
        return self._event_name

    @event_name.setter
    def event_name(self, var):
        try:
            var = str(var)
        except:
            raise TypeError("event_name not a string or data type that can be converted to string")

        self._event_name = var

    @property
    def event_country(self):
        """The country where the event occurred.

        Returns
        -------
        str
            The country where the event occurred.

        """
        return self._event_country

    @event_country.setter
    def event_country(self, var):
        try:
            var = str(var)
        except:
            raise TypeError("event_country not a string or data type that can be converted to string")

        self._event_country = var

    @property
    def date1(self):
        """Date 1

        Returns
        -------
        datetime.datetime
            The date and time
        """
        return self._date1

    @date1.setter
    def date1(self, date_time):
        if not isinstance(date_time, datetime):
            try:
                date_time = _date_time_from_json(date_time)
            except:
                raise TypeError(
                    "date_time must be instance of 'datetime.datetime' or a"
                    "json formatted date-time. Got {} instead".format(type(date_time))
                    )
        self._date1 = date_time

    @property
    def date2(self):
        """Date 2

        Returns
        -------
        datetime.datetime
            The date and time
        """
        return self._date2

    @date2.setter
    def date2(self, date_time):
        if not isinstance(date_time, datetime):
            try:
                date_time = _date_time_from_json(date_time)
            except:
                raise TypeError(
                    "date_time must be instance of 'datetime.datetime' or a"
                    "json formatted date-time. Got {} instead".format(type(date_time))
                    )
        self._date2 = date_time
    @property
    def processed_date(self):
        """Processed date

        Returns
        -------
        datetime.datetime
            The date and time
        """
        return self._processed_date

    @processed_date.setter
    def processed_date(self, date_time):
        if not isinstance(date_time, datetime):
            try:
                date_time = _date_time_from_json(date_time)
            except:
                raise TypeError(
                    "date_time must be instance of 'datetime.datetime' or a"
                    "json formatted date-time. Got {} instead".format(type(date_time))
                    )
        self._processed_date = date_time

    @property
    def event_date(self):
        """Event date

        Returns
        -------
        datetime.datetime
            The date and time
        """
        return self._processed_date

    @event_date.setter
    def event_date(self, date_time):
        if not isinstance(date_time, datetime):
            try:
                date_time = _date_time_from_json(date_time)
            except:
                raise TypeError(
                    "date_time must be instance of 'datetime.datetime' or a"
                    "json formatted date-time. Got {} instead".format(type(date_time))
                    )
        self._event_date = date_time

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
        """Plot interferogram

        Parameters
        ----------
        wrap: bool
            If ``True``, wrap the function
        ax: matplotlib.ax.Axes
            An axes object

        Returns
        -------
        matplotlib.ax.pcolormesh
            The inteferogram plot

        """

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
        """Plot masked interferogram

        Parameters
        ----------
        ax: matplotlib.ax.Axes
            An axes object
        opacity: float
            The opacity, default = 0.2

        Returns
        -------
        matplotlib.ax.pcolormesh
            The masked inteferogram plot

        """

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
        """calculate beta - the angle at earth center between reference point
        and satellite nadir

        Parameters
        ----------
        locations: list
            list of locations

        Returns
        -------
        numpy.ndarray
            The LOS vectors
        """
        if not isinstance(locations, list):
            locations = [locations]

        utmZone = self.location_UTM_zone
        refPoint = self.ref
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
            refPoint[0], refPoint[1], np.abs(utmZone), northern=utmZone > 0
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

        return np.squeeze(np.array([los_x, los_y, los_z]))

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


class Oksar:

    def __init__(
        self,
        O,
        U,
        V,
        center,
        depth_top,
        depth_bottom=1e4,
        strike=0.,
        dip=45.,
        rake=90,
        slip=0.5,
        length=1e4,
        beta=3e10,
        mu=3e10,
        shape=(300, 300)
    ):

        self.O = O
        self.U = U
        self.V = V
        self.center = center
        self.depth_top = depth_top
        self.depth_bottom = depth_bottom
        self.strike = strike
        self.dip = dip
        self.rake = rake
        self.slip = slip
        self.length = length
        self.beta = beta
        self.mu = mu
        self.shape = shape

    @property
    def O(self):
        """Origin of the simulation domain.

        Returns
        -------
        (2,) numpy.ndarray
            Origin of the simulation domain.

        """
        return self._O

    @O.setter
    def O(self, var):
        try:
            var = np.asarray(var, dtype=float)
        except:
            raise TypeError("O must be array_like of float")
        var = np.squeeze(var)
        if var.shape != (2, ):
            raise ValueError(f"O must have a shape (2, ) not {var.shape}")
        self._O = var

    @property
    def U(self):
        """U direction of the simulation domain

        Returns
        -------
        (2,) numpy.ndarray
            U direction of the simulation domain

        """
        return self._U

    @U.setter
    def U(self, var):
        try:
            var = np.asarray(var, dtype=float)
        except:
            raise TypeError(
                "U must be a array_like of float"
            )
        var = np.squeeze(var)
        if var.shape != (2, ):
            raise ValueError(f"U must have a shape of (2, ) not {var.shape}")
        self._U = var

    @property
    def V(self):
        """V direction of the simulation domain

        Returns
        -------
        (2,) numpy.ndarray
            V direction of the simulation domain

        """
        return self._location

    @V.setter
    def V(self, var):
        try:
            var = np.asarray(var, dtype=float)
        except:
            raise TypeError("V must be array_like of float")
        var = np.squeeze(var)
        if var.shape != (2, ):
            raise ValueError(f"V must have a shape of (2, ) not {var.shape}")
        self._V = var


    @property
    def center(self):
        """Earthquake epicenter (bottom N, left E).

        Returns
        -------
        )2,) numpy.ndarray
            Earthquake epicenter (bottom N, left E)

        """
        return self._center

    @center.setter
    def center(self, var):
        try:
            var = np.asarray(var, dtype=float)
        except:
            raise TypeError("center must be array_like of float")
        var = np.squeeze(var)
        if var.shape != (2, ):
            raise ValueError(f"center must have a shape of (2, ) not {var.shape}")
        self._center = var

    @property
    def depth_top(self):
        """Depth to the top of the fault (m)

        Returns
        -------
        float
            Depth to the top of the fault (m)

        """
        return self._depth_top

    @depth_top.setter
    def depth_top(self, var):
        try:
            var = float(var)
        except:
            raise TypeError("depth_top must be a float")

        if var < 0.:
            raise ValueError("depth_top must be equal or greather than 0")

        self._depth_top = var

    @property
    def depth_bottom(self):
        """Depth to the bottom of the fault (m)

        Returns
        -------
        float
            Depth to the bottom of the fault (m)

        """
        return self._depth_bottom

    @depth_bottom.setter
    def depth_bottom(self, var):
        try:
            var = float(var)
        except:
            raise TypeError("depth_bottom must be a float")

        if var < self.depth_top:
            raise ValueError("depth_bottom must be larger than depth_top")

        self._depth_bottom = var

    @property
    def strike(self):
        """Strike angle (0 to 360) in degrees

        Returns
        -------
        float
            Strike angle (0 to 360)

        """
        return self._strike

    @strike.setter
    def strike(self, var):
        try:
            var = float(var)
        except:
            raise TypeError("strike must be a float")

        if (var < 0.) | (var > 360.):
            raise ValueError("strike must be within [0, 360]")

        self._strike = var

    @property
    def dip(self):
        """Strike angle (0 to 90) in degrees

        Returns
        -------
        float
            Strike angle (0 to 90) in degrees

        """
        return self._dip

    @dip.setter
    def dip(self, var):
        try:
            var = float(var)
        except:
            raise TypeError("dip must be a float")

        if (var < 0.) | (var > 90.):
            raise ValueError("dip must be within [0, 90]")

        self._dip = var

    @property
    def rake(self):
        """Rake angle (-180 to 180)

        Returns
        -------
        float
            Rake angle (-180 to 180)

        """
        return self._rake

    @rake.setter
    def rake(self, var):
        try:
            var = float(var)
        except:
            raise TypeError("rake must be a float")

        if (var < -180) | (var > 180.):
            raise ValueError("rake must be within [-180, 180]")

        self._rake = var

    @property
    def slip(self):
        """Slip

        Returns
        -------
        float
            Slip

        """
        return self._slip

    @slip.setter
    def slip(self, var):
        try:
            var = float(var)
        except:
            raise TypeError("slip must be a float")

        if var < 0.:
            raise ValueError("slip must be greater than or equal to 0")

        self._slip = var

    @property
    def length(self):
        """Length

        Returns
        -------
        float
            Length

        """
        return self._length

    @length.setter
    def length(self, var):
        try:
            var = float(var)
        except:
            raise TypeError("length must be a float")

        if var < 0.:
            raise ValueError("length must be greater than or equal to 0")

        self._length = var

    @property
    def beta(self):
        """Beta parameter

        Returns
        -------
        float
            Beta parameter

        """
        return self._beta

    @beta.setter
    def beta(self, var):
        try:
            var = float(var)
        except:
            raise TypeError("beta must be a float")

        self._beta = var

    @property
    def mu(self):
        """Mu parameter

        Returns
        -------
        float
            Mu parameter

        """
        return self._mu

    @mu.setter
    def mu(self, var):
        try:
            var = float(var)
        except:
            raise TypeError("mu must be a float")

        self._mu = var

    @property
    def shape(self):
        """Shape of the interferogram defined by number of pixels in
        the North and Easting directions, respectively; i.e. (n_pix_N, n_pix_E).

        Returns
        -------
        (2) numpy.ndarray of int
            Interferogram shape.

        """
        return self._shape

    @shape.setter
    def shape(self, var):
        try:
            var = np.array(var, dtype=int)
        except:
            raise TypeError("shape must be a (2) array_like of int")

        if len(var) != 2:
            raise TypeError("shape must be a (2) array_like of int")

        self._shape = var

    @property
    def simulation_grid(self):
        """Compute simulation grid

        Returns
        -------
        numpy.ndarray
            Simulation grid
        """
        shape = self.shape
        origin = self.O
        R = np.stack([self.U, self.V])
        square = np.meshgrid(
            np.linspace(0, 1, shape[0]),
            np.linspace(0, 1, shape[1])
        )
        square = np.stack([square[0].flatten(), square[1].flatten()], axis=-1)
        grid = origin + np.einsum('ij,...j->...i', R, square)

        return grid

    @property
    def displacement_vector(self):
        """Compute displacement vector

        Returns
        -------
        numpy.ndarray
            Displacement vector

        """

        vec = self.simulation_grid
        x, y = vec[:, 0], vec[:, 1]

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

        if(hmin == 0.0):
            hmin = 0.00001

        sstrike = (strike+90.0)*DEG2RAD

        ct = np.cos(sstrike)
        st = np.sin(sstrike)

        X = ct * (-flt_x + x) - st * (-flt_y + y)
        Y = ct * (-flt_y + y) + st * (-flt_x + x)

        u = self._dc3d3(alpha, X, Y, -dip, al1, al2, aw1, aw2, us, ud)

        UX = ct*u[...,0] + st*u[...,1]
        UY = -st*u[...,0] + ct*u[...,1]
        UZ = u[..., 2]

        return np.stack([UX, UY, UZ], axis=-1)

    def _dc3d3(self, alpha, X, Y, dip, al1, al2, aw1, aw2, disl1, disl2):
        F0 = 0.0
        F1 = 1.0
        F2 = 2.0
        PI2 = 6.283185307179586
        EPS = 1.0E-6

        u = np.array([F0, F0, F0])
        dub = np.array([F0, F0, F0])

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
                    raise Exception('dccon2 b %f %f %f %f %f' % (
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
                    du2 = np.array([du2x, du2y, du2z])
                    dub = du2 * (disl1 / PI2)
                else:
                    dub = np.zeros(3)

                # dip-slip contribution
                if(disl2 != F0):
                    du2x = - q/c2_r + c0_alp3 * ai3 * c0_sdcd
                    du2y = - et*qx - c2_tt - c0_alp3 * xi / rd * c0_sdcd
                    du2z = q*qx + c0_alp3 * ai4 * c0_sdcd
                    du2 = np.array([du2x, du2y, du2z])
                    dub = dub + (du2 * (disl2 / PI2))

                # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                dux = dub[0]
                duy = dub[1]*c0_cd - dub[2]*c0_sd
                duz = dub[1]*c0_sd + dub[2]*c0_cd
                du = np.array([dux, duy, duz])
                if((j+k) != 3):
                    u = + du + u
                else:
                    u = - du + u
        return u

    def plot_displacement(self, eq=None, ax=None, wrap=True, mask_opacity=0.2):
        """Plot displacement/

        Parameters
        ----------
        eq: EarthquakeInterferogram
            Instance of ``EarthquakeInterferogram``
        ax: matplotlib.ax.Axes
            Axes object
        wrap: bool
            If ``True``, wrap the function
        mask_opacity: float
            Masking opacity, default = 0.2

        Returns
        -------
        matplotlib.ax.pcolormesh
            Displacement plot

        """
        if eq is not None and not isinstance(eq, EarthquakeInterferogram):
            raise TypeError("eq must be an EarthquakeInterferogram")

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
            LOS = np.array([0, 0, 1])
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

    # data = np.fromstring(dinar_file.content, np.float32)

    # dinar = EarthquakeInterferogram(
    #     # uid='dinar',
    #     title='Dinar, Turkey',
    #     description=(
    #         'On October 1, 1995, a strong earthquake ruptured a section of '
    #         'the Dinar-Civril fault in SW Turkey. Around 30% of the buildings '
    #         'in the nearby town of Dinar were destroyed. 92 inhabitants were '
    #         'killed and over 200 injured.'
    #     ),
    #     event_country='Turkey',
    #     event_date='1995-09-30T18:00:00Z',
    #     event_gcmt_id='100195B',
    #     event_name='Dinar',
    #     copyright='ESA',
    #     data_source='ESA',
    #     date1='1995-08-12T18:00:00Z',
    #     date2='1995-12-31T17:00:00Z',
    #     processed_by='GarethFunning',
    #     processed_date='2003-01-20T17:00:00Z',
    #     ref_incidence=23,
    #     ref=[741140., 4230327.],
    #     scaling=0.0045040848895,
    #     local_earth_radius=6386232,
    #     local_rigidity=30000000000,
    #     location=[706216.0606, 4269238.9999],
    #     shape=(1024, 1024),
    #     location_UTM_zone=35,
    #     pixel_size=[80., 80.],
    #     satellite_altitude=788792,
    #     satellite_azimuth=192,
    #     satellite_fringe_interval=0.028333333,
    #     satellite_name='ERS',
    #     data=data
    # )

    data = np.fromstring(dinar_file.content, np.float32)
    title = 'Dinar, Turkey'
    location = [706216.0606, 4269238.9999]
    location_UTM_zone = 35
    shape = (1024, 1024)
    pixel_size = [80., 80.]
    ref= [741140., 4230327.]
    ref_incidence = 23
    satellite_azimuth=192
    satellite_altitude=788792
    processed_by='GarethFunning'

    dinar = EarthquakeInterferogram(
        data,
        title,
        location,
        location_UTM_zone,
        shape,
        pixel_size,
        ref,
        ref_incidence,
        satellite_azimuth,
        satellite_altitude,
        processed_by,
        scaling=0.0045040848895,
        satellite_fringe_interval=0.028333333,
        local_earth_radius=6386232,
        local_rigidity=30000000000,
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
        processed_date='2003-01-20T17:00:00Z',
        satellite_name='ERS',
    )

    O = [706216.0606, 4187318.9999]
    U = [81920, 0]
    V = [0, 81920]
    center = [773728.2977967655, 4223586.816611591]
    depth_top=0

    dinar_fwd = Oksar(
        O,
        U,
        V,
        center,
        depth_top,
        depth_bottom=1.5e4,
        strike=329.6,
        dip=50.,
        rake=90,
        slip=0.5,
        length=11578.907244622129,
        beta=3e10,
        mu=3e10,
        shape=(300, 200)
    )

    return dinar, dinar_fwd
