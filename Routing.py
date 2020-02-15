###############################################################################################
#
# Routing library with routines to
# - read NetCDF GRIB file and show statistics using netcdf engine
# - provide TWS and TWD at a time , lat and long
# - calculate BSP from TWS and TWA
#   https: // www.movable - type.co.uk / scripts / latlong.html
#   https://gis.stackexchange.com/questions/5821/calculating-latitude-longitude-x-miles-from-point
#
# Ruud Kapteijn, 9-feb-2020
###############################################################################################

import numpy as np
import xarray as xr
from math import sin, asin, cos, sqrt, atan2, radians, degrees, pi


class GribIndexException(Exception):
    """ Exception subclass to throw for index out of range of GRIB file (lat, long or time)

    constructor parameters: message - string - error message
    """

    def __init__(self, message):
        self.message = message
# GribIndexException

class PolarException(Exception):
    """ Exception subclass to throw for error in polar class

    constructor parameters: message - string - error message
    """

    def __init__(self, message):
        self.message = message
# PolarException

class WayPoint(object):
    """ Waypoint object 2D position with latitude and longitude

    TODO: add method to return coordinates in format deg min.dec (string)
    TODO: add get_btw(wpt) method
    """

    def __init__(self, name, lat, long):
        """ Waypoint constructor - creates waypoint

        :param name: string - name of the waypoint
        :param lat:  float  - latitude of the waypoint as degrees.decimal degrees or string - converted with dec_crd()
        :param long: float  - longitude of the waypoint as degrees.decimal degrees or string - converted with dec_crd()
        """
        if type(lat) is str:
            self.lat = self.dec_crd(lat)
        else:
            self.lat = lat
        if type(long) is str:
            self.long = self.dec_crd(long)
        else:
            self.long = long
        self.name = name

    def dec_crd(self, s):
        """ decimal_coordinate transforms coordinate string deg min.dec to float

        :param s: coordinate in format "52° 34,777'"
        :return: coordinate as float (degrees.decimal degrees)
        """
        c = s.split(" ")
        # print("c[0]: -%s-" % (c[0]))
        c0 = c[0][0:len(c[0]) - 1]
        # c0 = c[0][0:len(c[0]) - 2]               # changed -1 to -2 for UTF-8 character (2 bytes)
        # print("c0: -%s-" % (c0))
        c1 = c[1][0:len(c[1]) - 1]
        c1 = c1.replace(",", ".")
        # print("c1: -%s-" % (c1))
        return float(c0) + float(c1) / 60

    def dtw(self, wpt):
        """distance to waypoint calculates distance from self waypoint to wpt

        :param wpt: waypoint to calculate distance to
        :return: float - distance in nautical miles
        """
        return DTW(self.lat, self.long, wpt.get_lat(), wpt.get_long())

    def btw(self, wpt):
        """bearing to waypoint calculates bearing from self waypoint to wpt

        :param wpt: waypoint to calculate bearing to
        :return: int - bearing in degrees
        """
        return BTW(self.lat, self.long, wpt.get_lat(), wpt.get_long())

    def get_lat(self):
        """ get_latitude

        :return: latitude of waypoint as float - degrees.decimal degrees
        """
        return self.lat

    def get_long(self):
        """ get_longitude

        :return: longitude of waypoint as float - degrees.decimal degrees
        """
        return (self.long)

# end WayPoint

class Route(object):
    """Route - series of waypoints

    """

    def __init__(self, name):
        """ Contructor for Route

        after creation the waypoint list is empty
        :param name: string - name of the route
        """
        self.name = name
        self.wpt_list = []

    def add_wpt(self, wpt):
        """ add waypoint to the end of the route

        :param wpt: Waypoint - object of class waypoint
        :return: na
        """
        self.wpt_list.append(wpt)

    def wpt(self, i):
        """return i-th waypoint of the waypoint list

        :param i: int index should be between 0 and len(wpt_list)
        :return: Waypoint - object of class waypoint
        """
        return self.wpt_list[i]

    def dst(self):
        """ distance calculate length of route in nm

        :return: float - length of route in nm
        """
        _dist = 0.0
        for i in range(0, len(self.wpt_list) - 1):
            _dist += self.wpt_list[i].dtw(self.wpt_list[i + 1])
        return _dist

    def wpt_cnt(self):
        """ return numer of waypoint in the route

        :return: int - number of waypoints in the route
        """
        return len(self.wpt_list)

    def stats(self):
        """statistics - return string of statistics

        :return: string - statistics of route
        """
        res = "Route %s, %d waypoints, length: %5.2f nm" % (self.name, self.wpt_cnt(), self.dst())
        if len(self.wpt_list) > 2:
            res = res + " (only first two waypoints taken into account for routing)"
        return res
# end Route

class TrackPoint(object):
    """TrackPoint object
    """

    def __init__(self, lat, long, time):
        """contructor of trackpoint object

        :param lat: float - latitude of trackpoint as degrees.dec degr
        :param long: float - longitude of trackpoint as degrees.dec degr
        :param time: np.datetime64 - time of trackpoint
        """
        self.lat = lat
        self.long = long
        self.time = time

    def get_lat(self):
        """ get latitude of trackpoint

        :return: float - return latitude of trackpoint as degrees.dec deg
        """
        return self.lat

    def get_long(self):
        """ get longitude of trackpoint

        :return: float - return longitude of trackpoint as degrees.dec deg
        """
        return (self.long)

    def get_wpt(self):
        """ return latitude and longitude of trackpoint as Waypoint object

        :return: waypoint object - location of track point
        """
        _wpt = WayPoint(name="tmp", lat=self.lat, long=self.long)
        return _wpt

    def get_time(self):
        """get time of trackpoint

        :return: np.datetime64 - time of track point
        """
        return self.time

# end TrackPoint

class WindObject(object):
    """Encapsulates GRIB file (in NETCDF format) and provides wind data
    """

    def __init__(self, filename):
        """ WindObject constuctor create wind object

        :param filename: string - name of NETCDF file containing GRIB data
        """
        self.ds = xr.open_dataset(filename, engine='netcdf4')
        self.timesteps = len(self.ds.step.values)
        self.timemin = self.ds.valid_time.values[0]                     # type is numpy.datetime64
        self.timemax = self.ds.valid_time.values[self.timesteps - 1]    # type is numpy.datetime64
        self.timestep = np.timedelta64(self.timemax - self.timemin) / self.timesteps
        self.latsteps = len(self.ds.latitude.values)
        self.latmin = self.ds.latitude.values[0]                        # type is ?
        self.latmax = self.ds.latitude.values[self.latsteps - 1]        # type is ?
        self.latstep = (self.latmax - self.latmin) / self.latsteps
        self.longsteps = len(self.ds.longitude.values)
        self.longmin = self.ds.longitude.values[0]                      # type is ?
        self.longmax = self.ds.longitude.values[self.longsteps - 1]     # type is ?
        self.longstep = (self.longmax - self.longmin) / self.longsteps

    def get_stats(self):
        """ get_statistics show contents of GRIB file

        :return: string - printable description string
        """
        return str(self.ds)

    def _get_wind(self, tpt):
        """ _get_wind private method to calculate true wind direction and true wind speed at lat, long & time

        :param tpt: track_point with lat, long & time to calculate wind for
        :return: dict with tws: tws as float, twd: twd as float
        """
        time = tpt.get_time()
        lat = tpt.get_lat()
        long = tpt.get_long()
        if time < self.timemin or time > self.timemax:
            raise GribIndexException("GRIB time out of range, index: %s, min: %s, max: %s" % (
            str(time), str(self.timemin), str(self.timemax)))
        if lat < self.latmax or lat > self.latmin:  # latitude coordinates are inverted
            raise GribIndexException("GRIB latitude out of range, index: %s, min: %s, max: %s" % (
            str(lat), str(self.latmax), str(self.latmin)))
        if long < self.longmin or long > self.longmax:
            raise GribIndexException("GRIB longitude out of range, index: %s, min: %s, max: %s" % (
            str(long), str(self.longmin), str(self.longmax)))

        _timeindex = np.rint((time - self.timemin) / self.timestep).astype(int)
        _latindex = np.rint((lat - self.latmin) / self.latstep).astype(int)
        _longindex = np.rint((long - self.longmin) / self.longstep).astype(int)

        u10 = self.ds.u10.values[_timeindex, _latindex, _longindex]
        v10 = self.ds.v10.values[_timeindex, _latindex, _longindex]

        return {'tws': sqrt(u10 ** 2 + v10 ** 2) * 3600 / 1852, 'twd': atan2(u10, v10) * 180 / pi + 180}
    # _get_wind()

    def get_tws(self, tpt):
        """ public method to get true wind speed

        :param tpt: trackpoint with latitude, longitude and time
        :return: float - true wind speed in knots
        """

        return self._get_wind(tpt)['tws']
    # get_tws()

    def get_twd(self, tpt):
        """ public method to get true wind direction

        :param tpt: trackpoint with latitude, longitude and time
        :return: int - true wind direction in degrees
        """
        return self._get_wind(tpt)['twd']
    # get_tws()

# end of WindObject

class PolarObject(object):
    """Encapsulates Polar file (in Expedition format) and provides boat speed  for wind speed and angle
    """

    def __init__(self, filename):
        """ PolarObject constructor - reads polar file and prepares data structures

        file is tab delimited text file
        each row starts with TWS followed by TWA - BSP combinations - no of cols is odd
        TWS should be sorted from low to high
        each row should have the same number of TWA - BSP combinations
        TWA - BSP combinations should be sorted by TWA low to high
        Columns should have similar TWA's not necessarily the same
        Optimal beat angle and optimal run angle should be included in the TWA's

        reads file into into numpy.ndarray, structure cf. Expedition (tab delimited txt file)
        creates:
        self.tws_vector - a list of TWS values as float
        self.twa_array  - a 2D array of TWA values as floats
        self.bsp_array  - a 2D array of BSP values as floats
        :param filename: string - name of polar file (polar.txt)
        """

        self.polar = np.loadtxt(filename, delimiter='\t', dtype=np.float32)

        # create list of TWS values
        self.tws_vector = []
        for i in range(0, self.polar.shape[0]):
            self.tws_vector.append(self.polar[i][0])
        self.TWS = self.tws_vector.copy()
        
        # create 2D array of TWA values
        self.twa_array = self.polar.copy()
        self.twa_array = np.delete(self.twa_array, 0, 1)  # delete 1st column with windspeeds
        self.twa_array = np.delete(self.twa_array, 0, 1)  # delete 2nd column (new first) with TWA's = 0
        self.twa_array = np.delete(self.twa_array, 0, 1)  # delete 3rd column (new first) with BSP's at TWA = 0
        cols = self.twa_array.shape[1] / 2
        i = 1
        while self.twa_array.shape[1] > cols:
            self.twa_array = np.delete(self.twa_array, i, 1)
            i += 1

        # create 2D array of BSP values
        self.bsp_array = self.polar.copy()
        self.bsp_array = np.delete(self.bsp_array, 0, 1)  # delete 1st column with windspeeds
        self.bsp_array = np.delete(self.bsp_array, 0, 1)  # delete 2nd column (new first) with TWA's = 0
        self.bsp_array = np.delete(self.bsp_array, 0, 1)  # delete 3rd column (new first) with BSP's at TWA = 0
        self.bsp_array = np.delete(self.bsp_array, 0, 1)  # delete 4rd column (new first) with TWA's (likely 30)
        i = 1
        while self.bsp_array.shape[1] > cols:
            self.bsp_array = np.delete(self.bsp_array, i, 1)
            i += 1

    def print_stats(self):
        """ print statistics of polar file

        :return: string - description of polar file
        """

        print("TWS: %s" % str(self.TWS))
        print("TWA: %s" % str(self.TWA))
        print("polar shape: %s" % str(self.polar.shape))
        print(self.polar)
        # print("polar[0,0]: %5.2f" % self.polar.item((0, 0)))

    def get_twa_vector(self, tws):
        """ get vector of TWA's for given TWS does interpolation between TWS values

        :param tws: float - true wind speed for which vector is calculated in knots
        :return: list of true wind angles - floats
        """

        # if tws >= max wind in polar return max wind vector
        if tws >= self.tws_vector[-1]:
            return self.twa_array[-1,]

        # if tws < min wind in polar return interpolation with 0
        if tws < self.tws_vector[0]:
            f = 1 - (self.tws_vector[0] - tws) / self.tws_vector[0]
            return self.twa_array[0,] * f

        # find wind interval in TWS
        for i in range(0, len(self.tws_vector) - 1):
            # if tws == TWS no interpolation required
            if tws == self.tws_vector[i]:
                return self.twa_array[i,]
            # interpolate between TWS[i] and TWS[i+1]
            if tws > self.tws_vector[i] and tws < self.tws_vector[i + 1]:
                d = self.tws_vector[i + 1] - self.tws_vector[i]
                f1 = 1 - (tws - self.tws_vector[i]) / d
                f2 = 1 - (self.tws_vector[i + 1] - tws) / d
                return self.twa_array[i,] * f1 + self.twa_array[i + 1,] * f2
        raise PolarException("no true wind angle vector calculated")

    def _get_bsp_vector(self, tws):
        """ get vector of BSP's for given TWS does interpolation between TWS values

        :param tws: float - true wind speed for which vector is calculated in knots
        :return: list of boat speeds in knots - floats
        """

        # if tws >= max wind in polar return max wind vector
        if tws >= self.tws_vector[-1]:
            return self.bsp_array[-1,]

        # if tws < min wind in polar return interpolation with 0
        if tws < self.tws_vector[0]:
            f = 1 - (self.tws_vector[0] - tws) / self.tws_vector[0]
            return self.bsp_array[0,] * f

        # find wind interval in TWS
        for i in range(0, len(self.tws_vector) - 1):
            # if tws == TWS no interpolation required
            if tws == self.tws_vector[i]:
                return self.bsp_array[i,]
            # interpolate between TWS[i] and TWS[i+1]
            if tws > self.tws_vector[i] and tws < self.tws_vector[i + 1]:
                d = self.tws_vector[i + 1] - self.tws_vector[i]
                f1 = 1 - (tws - self.tws_vector[i]) / d
                f2 = 1 - (self.tws_vector[i + 1] - tws) / d
                return self.bsp_array[i,] * f1 + self.bsp_array[i + 1,] * f2
        raise PolarException("no boat speed vector calculated")

    def get_bsp(self, tws, twa):
        """ get boat speed at given truw wind speed and true wind angle

        Raises PolarException iif no boat speed can be calculated
        :param tws: float - true wind speed in knots
        :param twa: int - true wind angle in degrees
        :return: float - boat speed in knots
        """

        twa= abs(twa)
        if tws < 0 or tws > 100 or twa < 0 or twa > 180:
            raise PolarException("PolarObject get_bsp(tws, twa) TWS or abs(TWA) out of range (%5.2f, %d)" % (tws, twa))

        twa_vector = self.get_twa_vector(tws)
        bsp_vector = self._get_bsp_vector(tws)
        if twa < twa_vector[0]:  # is closer to wind than minimum TWA in
            return 0.0
        else:
            # find twa interval in TWA
            for i in range(0, len(twa_vector)):
                # if twa == TWA no interpolation required
                if twa == twa_vector[i]:
                    return bsp_vector[i]
                # interpolate between TWA[i] and TWA[i+1]
                if twa > twa_vector[i] and twa < twa_vector[i + 1]:
                    d = twa_vector[i + 1] - twa_vector[i]
                    f1 = 1 - (twa - twa_vector[i]) / d
                    f2 = 1 - (twa_vector[i + 1] - twa) / d
                    return bsp_vector[i] * f1 + bsp_vector[i + 1] * f2
            raise PolarException("no wind value calculated")

# end PolarObject

class Itinerary(object):

    #    def __init__(self, route, start_time):       # constructor with route and start time
    def __init__(self, route, start_time):
        _org = route.wpt(0)
        self.dest = route.wpt(1)
        self.tpt_list = []  # track point list (track point = way point + time)
        tpt = TrackPoint(_org.get_lat(), _org.get_long(), start_time)
        self.tpt_list.append(tpt)

    def last_tpt(self):
        return self.tpt_list[-1]

    def add_tpt(self, lat, long, time):
        tpt = TrackPoint(lat, long, time)
        self.tpt_list.append(tpt)

    def get_dest(self):
        """get destination as Waypoint object

        :return: Waypoint object - destination
        """
        return self.dest

    def dist_to_go(self):
        lat1 = self.last_tpt().get_lat()
        long1 = self.last_tpt().get_long()
        lat2 = self.dest.get_lat()
        long2 = self.dest.get_long()
        return DTW(lat1, long1, lat2, long2)
# end Itinerary

def TWA(twd, hea):
    """calculates true wind angle for a given wind direction and heading

    :param twd: int - true wind direction in degrees
    :param hea: int - boat heading in degrees
    :return: int - true wind angle between heading and twd, negative is port, positive is starboard
    """

    r = twd - hea
    if r > -179 and r <= 180:
        # print("TWA(1. twd=%d, hea=%d, res=%d)" % (twd, hea, r))
        return r
    if r == -180:
        # print("TWA(2. twd=%d, hea=%d, res=180)" % (twd, hea))
        return 180
    if r > 180:
        # print("TWA(3. twd=%d, hea=%d, res=%d)" % (twd, hea, r - 360))
        return r - 360
    if r < -180:
        # print("TWA(4. twd=%d, hea=%d, res=%d)" % (twd, hea, 360 + r))
        return 360 + r
# TWA()

def DTW(lat1, long1, lat2, long2):
    # Haversine
    R = 6371 / 1.852
    _lat1, _long1, _lat2, _long2 = map(radians, [lat1, long1, lat2, long2])
    _dlong = _long2 - _long1
    _dlat = _lat2 - _lat1
    a = sin(_dlat / 2) ** 2 + cos(_lat1) * cos(_lat2) * sin(_dlong / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c
# DTW()

def BTW(lat1, long1, lat2, long2):
    _lat1, _long1, _lat2, _long2 = map(radians, [lat1, long1, lat2, long2])
    _dlong = _long2 - _long1
    y = sin(_dlong) * cos(_lat2)
    x = cos(_lat1) * sin(_lat2) - sin(_lat1) * cos(_lat2) * cos(_dlong)
    return (degrees(atan2(y, x)) + 360) % 360  # fmod((degrees(atan2(y, x)) + 360.0), 360.0)
# BTW()

def new_crd(lat, long, hea, dist):
    """ new_coordinate - calculate new coordinate based on current coordinate, heading and distance

    :param lat: float - latitude of current position deg.dec deg
    :param long: float - longitude of current position deg. dec deg
    :param hea: int heading to new position in degrees
    :param dist: float distance to new position in nm
    :return: dict with 'lat': latitude (float) and 'long': longitude (float)
    """
    R = 6371 / 1.852
    _lat, _long = map(radians, [lat, long])
    _hea = radians(hea)
    _lat2 = asin(sin(_lat) * cos(dist / R) + cos(_lat) * sin(dist / R) * cos(_hea))
    _long2 = _long + atan2(sin(_hea) * sin(dist / R) * cos(_lat), cos(dist / R) - sin(_lat) * sin(_lat2))
    return {'lat': degrees(_lat2), 'long': degrees(_long2)}

# marks = {"EA1"    :["52° 34,777'", "5° 13,433'"], \
#          "KG10"   :["52° 40,656'", "5° 16,039'"], \
#          "SPT-I"  :["52° 31,998'", "5° 21,390'"], \
#          "SPT-H"  :["52° 31,500'", "5° 7,999'"], \
#          "SPT-G"  :["52° 36,496'", "5° 8,000'"]}

################################ that's all folks ###########################################################
