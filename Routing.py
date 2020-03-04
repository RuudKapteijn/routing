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
import xml.etree.ElementTree as et

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

    def get_name(self):
        """ return name of the waypoint

        :return: string - name of the waypoint
        """
        return self.name

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

    def __init__(self, name, gpxfile = None):
        """ Contructor for Route

        if gpxfile is not supplied after creation the waypoint list is empty
        if gpxfile is supplied this is considered the name of a gpx (XML) file with route data
        route is created from file
        :param name: string - name of the route
        :param gpxfile: string - name of gpx file containing route data
        """
        self.name = name
        self.wpt_list = []

        if gpxfile:
            # open XML tree from the input file
            tree = et.parse(gpxfile)
            root = tree.getroot()
            # create name space dict
            ns = {'gpx': 'http://www.topografix.com/GPX/1/1'}
            # find first route cq. first element with tag 'rte' (additional route tags are ignored)
            route = root.find('gpx:rte', ns)
            # if found rte tag cq. route != None
            if route:
                self.name = route.find('gpx:name', ns).text         # tag contains route name - over write name
                # find all rtept's (route points) in route
                for routepoint in route.findall('gpx:rtept', ns):
                    wp = WayPoint(name=routepoint.find('gpx:name', ns).text, lat=float(routepoint.attrib['lat']),\
                                  long=float(routepoint.attrib['lon']))
                    self.wpt_list.append(wp)

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
        res = "Route %s, %d waypoints, length: %5.2f nm, 1st leg (%s - %s) bearing: %d gr, distance: %5.2f" % \
              (self.name, self.wpt_cnt(), self.dst(), self.wpt(0).get_name(), self.wpt(1).get_name(), \
               self.wpt(0).btw(self.wpt(1)), self.wpt(0).dtw(self.wpt(1)))
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
    """Encapsulates GRIB files for wind and current

     (in NETCDF format) and provides wind data
     wind files from Squid
     Traditionally, set and drift is defined as the effect a current has on a vessel's movement,
     with set referring to direction and drift to speed
     current files from: http://www.sailingweatheronline.com/bsh_currents.html
    """

    def __init__(self, windfile, currentfile=None):
        """ WindObject constuctor create wind object

        :param filename: string - name of NETCDF file containing GRIB data
        """
        print("Load wind data from %s" % windfile)
        self.w_ds = xr.open_dataset(windfile, engine='netcdf4')
        self.w_timesteps = len(self.w_ds.time.values)
        self.w_timemin = self.w_ds.time.values[0]                       # type is numpy.datetime64
        self.w_timemax = self.w_ds.time.values[-1]                      # type is numpy.datetime64
        self.w_timestep = np.timedelta64(self.w_timemax - self.w_timemin) / self.w_timesteps
        self.w_latsteps = len(self.w_ds.latitude.values)
        self.w_latmin = self.w_ds.latitude.values[0]                    # type is ?
        self.w_latmax = self.w_ds.latitude.values[-1]                   # type is ?
        self.w_latstep = (self.w_latmax - self.w_latmin) / self.w_latsteps
        self.w_longsteps = len(self.w_ds.longitude.values)
        self.w_longmin = self.w_ds.longitude.values[0]                  # type is ?
        self.w_longmax = self.w_ds.longitude.values[-1]                 # type is ?
        self.w_longstep = (self.w_longmax - self.w_longmin) / self.w_longsteps

        if currentfile:
            print("Load currents from: %s" % currentfile)
            self.c_ds = xr.open_dataset(currentfile, engine='netcdf4')
            self.c_timesteps = len(self.c_ds.time.values)
            self.c_timemin = self.c_ds.time.values[0]                       # type is numpy.datetime64
            self.c_timemax = self.c_ds.time.values[-1]                      # type is numpy.datetime64
            self.c_timestep = np.timedelta64(self.c_timemax - self.c_timemin) / self.c_timesteps
            self.c_latsteps = len(self.c_ds.latitude.values)
            self.c_latmin = self.c_ds.latitude.values[0]                    # type is ?
            self.c_latmax = self.c_ds.latitude.values[-1]                   # type is ?
            self.c_latstep = (self.c_latmax - self.c_latmin) / self.c_latsteps
            self.c_longsteps = len(self.c_ds.longitude.values)
            self.c_longmin = self.c_ds.longitude.values[0]                  # type is ?
            self.c_longmax = self.c_ds.longitude.values[-1]                 # type is ?
            self.c_longstep = (self.c_longmax - self.c_longmin) / self.c_longsteps
            self.current = True
        else:
            self.current = False
            print("No current data available")

    def get_stats(self):
        """ get_statistics show contents of GRIB file

        :return: string - printable description string
        """
        return str(self.w_ds)

    def _get_ground_wind(self, tpt):
        """ _get_ground_wind private method to calculate true wind direction and true wind speed at lat, long & time

        return ground wind without taking current into account (wind at anchor)
        :param tpt: track_point with lat, long & time to calculate wind for
        :return: dict with tws: tws as float, twd: twd as float
        """
        time = tpt.get_time()
        lat = tpt.get_lat()
        long = tpt.get_long()
        if time < self.w_timemin or time > self.w_timemax:
            raise GribIndexException("Wind GRIB time out of range, index: %s, min: %s, max: %s" % (
                str(time), str(self.w_timemin), str(self.w_timemax)))
        if lat < self.w_latmin or lat > self.w_latmax:
            raise GribIndexException("Wind GRIB latitude out of range, index: %s, min: %s, max: %s" % (
                str(lat), str(self.w_latmin), str(self.w_latmax)))
        if long < self.w_longmin or long > self.w_longmax:
            raise GribIndexException("Wind GRIB longitude out of range, index: %s, min: %s, max: %s" % (
                str(long), str(self.w_longmin), str(self.w_longmax)))

        _timeindex = np.rint((time - self.w_timemin) / self.w_timestep).astype(int)
        _latindex = np.rint((lat - self.w_latmin) / self.w_latstep).astype(int)
        _longindex = np.rint((long - self.w_longmin) / self.w_longstep).astype(int)
        try:
            u10 = self.w_ds.UGRD_10maboveground.values[_timeindex, _latindex, _longindex]
            v10 = self.w_ds.VGRD_10maboveground.values[_timeindex, _latindex, _longindex]
        except IndexError:
            raise GribIndexException(f"Wind GRIB Index Exception\ntime: {self.w_timemin} - {time} - {self.w_timemax}\n"
                                     f"lat: {self.w_latmin} - {lat} - {self.w_latmax}, long: time: {self.w_longmin} - {long} - {self.w_longmax}\n")
        return {'tws': sqrt(u10**2 + v10**2) * 3600/1852, 'twd': degrees(atan2(u10, v10)) % 360, 'hor': u10, 'ver': v10}
    # _get_wind()

    def _get_current(self, tpt):
        """ _get_current private method to calculate true wind direction and true wind speed at lat, long & time

        :param tpt: track_point with lat, long & time to calculate wind for
        :return: dict with tws: tws as float, twd: twd as float
        """
        time = tpt.get_time()
        lat = tpt.get_lat()
        long = tpt.get_long()
        if time < self.c_timemin or time > self.c_timemax:
            raise GribIndexException("Current GRIB time out of range, index: %s, min: %s, max: %s" % (
                str(time), str(self.c_timemin), str(self.c_timemax)))
        if lat < self.c_latmin or lat > self.c_latmax:
            raise GribIndexException("Current GRIB latitude out of range, index: %s, min: %s, max: %s" % (
                str(lat), str(self.c_latmin), str(self.c_latmax)))
        if long < self.c_longmin or long > self.c_longmax:
            raise GribIndexException("Current GRIB longitude out of range, index: %s, min: %s, max: %s" % (
                str(long), str(self.c_longmin), str(self.c_longmax)))

        _timeindex = np.rint((time - self.c_timemin) / self.c_timestep).astype(int)
        _latindex = np.rint((lat - self.c_latmin) / self.c_latstep).astype(int)
        _longindex = np.rint((long - self.c_longmin) / self.c_longstep).astype(int)
        print(f"timeindex: {_timeindex}, latindex: {_latindex}, longindex: {_longindex}")
        try:
            u10 = self.c_ds.UOGRD_2mbelowsealevel.values[_timeindex, _latindex, _longindex]
            v10 = self.c_ds.VOGRD_2mbelowsealevel.values[_timeindex, _latindex, _longindex]
        except IndexError:
            print("Current GRIB IndexException")
            raise GribIndexException(f"Current GRIB Index Exception\ntime: {self.c_timemin} - {time} - {self.c_timemax}\n"
                                     f"lat: {self.c_latmin} - {lat} - {self.c_latmax}, long: time: {self.c_longmin} - {long} - {self.c_longmax}\n")

        if np.isnan(u10) or np.isnan(v10):
            print(f"Current retrieval error, u10: {u10}, v10: {v10}")
            u10 = 0.0
            v10 = 0.0
        _dft = sqrt(u10**2 + v10**2) * 3600/1852
        _set = degrees(atan2(u10, v10)) % 360
        if np.isnan(_dft) or np.isnan(_set):
            print(f"Current retrieval error, _dft: {_dft}, _set: {_set}")

        return {'dft': _dft, 'set': _set, 'hor': u10, 'ver': v10}
    # _get_current()

    def get_tws(self, tpt):
        """ public method to get true wind speed

        :param tpt: trackpoint with latitude, longitude and time
        :return: float - true wind speed in knots
        """
        if not self.current:
            # print(f"get_tws: {self._get_ground_wind(tpt)['tws']}")
            return self._get_ground_wind(tpt)['tws']
        else:
            res_vector = sum_vectors(self._get_ground_wind(tpt)['twd'], self._get_ground_wind(tpt)['tws'], \
                                     opposite_direction(self._get_current(tpt)['set']), self._get_current(tpt)['dft'])
            return res_vector['len']
    # get_tws()

    def get_twd(self, tpt):
        """ public method to get true wind direction

        :param tpt: trackpoint with latitude, longitude and time
        :return: int - true wind direction in degrees
        """
        if not self.current:
            return self._get_ground_wind(tpt)['twd']
        else:
            _gwd = self._get_ground_wind(tpt)['twd']
            _gws = self._get_ground_wind(tpt)['tws']
            _set = self._get_current(tpt)['set']
            _dft = self._get_current(tpt)['dft']
            res_vector = sum_vectors(_gwd, _gws, opposite_direction(_set), _dft)
            print(f"_gwd: {_gwd}, _gws: {_gws}, _set: {_set}, _dft: {_dft}, twd: {res_vector['dir']}")
            return res_vector['dir']
    # get_twd()

    def current(self):
        return self.current

    def get_set(self, tpt):
        return self._get_current(tpt)['set']

    def get_dft(self, tpt):
        return self._get_current(tpt)['dft']

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
        print(f"Load polar data from {filename}")
        # create list of TWS values
        self.tws_vector = []
        for i in range(0, self.polar.shape[0]):
            self.tws_vector.append(self.polar[i][0])
        self.TWS = self.tws_vector.copy()
        
        # create 2D array of TWA values
        self.twa_array = self.polar.copy()
        self.twa_array = np.delete(arr=self.twa_array, obj=0, axis=1)  # delete 1st column with windspeeds
        # check if first two columns only contain zero's - if so, remove
        if round(np.sum(a=self.twa_array, axis=0)[0]) == 0 and round(np.sum(a=self.twa_array, axis=0)[1]) == 0:
            self.twa_array = np.delete(arr=self.twa_array, obj=0, axis=1)  # delete 2nd col (now 1st) with TWA's = 0
            self.twa_array = np.delete(arr=self.twa_array, obj=0, axis=1)  # delete 3rd col (now 1st) with BSP's = 0
        cols = self.twa_array.shape[1] / 2                             # number of twa, bsp pairs remaining
        i = 1
        while self.twa_array.shape[1] > cols:                          # remove the bsp column, keep the twa columns
            self.twa_array = np.delete(arr=self.twa_array, obj=i, axis=1)
            i += 1

        # create 2D array of BSP values
        self.bsp_array = self.polar.copy()
        self.bsp_array = np.delete(arr=self.bsp_array, obj=0, axis=1)  # delete 1st column with windspeeds
        # check if first two columns only contain zero's - if so, remove
        if round(np.sum(a=self.bsp_array, axis=0)[0]) == 0 and round(np.sum(a=self.bsp_array, axis=0)[1]) == 0:
            self.bsp_array = np.delete(arr=self.bsp_array, obj=0, axis=1)  # delete 2nd col (now 1st) with TWA's = 0
            self.bsp_array = np.delete(arr=self.bsp_array, obj=0, axis=1)  # delete 3rd col (now 1st) with BSP's = 0
        self.bsp_array = np.delete(arr=self.bsp_array, obj=0, axis=1)  # delete 4rd column (now 1st) TWA's (likely 30)
        i = 1
        while self.bsp_array.shape[1] > cols:                          # remove twa columns, keep bsp columns
            self.bsp_array = np.delete(self.bsp_array, i, 1)
            i += 1

        if self.bsp_array.shape[0] != self.twa_array.shape[0] or self.bsp_array.shape[1] != self.twa_array.shape[1]:
            raise PolarException("TWA matrix and BSP matrix do not have the same dimensions")

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

        # if tws >= max wind in polar return max wind twa vector
        if tws >= self.tws_vector[-1]:
            return self.twa_array[-1,]

        # if tws < min wind in polar return min wind twa vector
        if tws < self.tws_vector[0]:
            return self.twa_array[0,]

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
                twa_vector = self.twa_array[i,] * f1 + self.twa_array[i + 1,] * f2
                # print(f"get_twa_vector, tws: {tws}, i: {i},  f1: {f1}, f2: {f2}, twa_vector: {str(twa_vector)}")
                return twa_vector
        raise PolarException("no true wind angle vector calculated")

    def _get_bsp_vector(self, tws):
        """ get vector of BSP's for given TWS does interpolation between TWS values

        :param tws: float - true wind speed for which vector is calculated in knots
        :return: list of boat speeds in knots - floats
        """

        # if tws >= max wind in polar return max wind vector
        # print(f"_get_bsp_vector, tws: {tws}, tws_vector: {str(self.tws_vector)}")
        if tws >= self.tws_vector[-1]:
            return self.bsp_array[-1,]

        # if tws < min wind in polar return interpolation with 0
        if tws < self.tws_vector[0]:
            f = 1 - (self.tws_vector[0] - tws) / self.tws_vector[0]
            _bsp_array = self.bsp_array[0,] * f
            # print(f"f: {f}, _bsp_array: {str(_bsp_array)}")
            return _bsp_array

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
        # print(f"get_bsp, tws: {tws}, twa: {twa}")
        twa= abs(twa)
        if tws < 0 or tws > 100 or twa < 0 or twa > 180:
            raise PolarException("PolarObject get_bsp(tws, twa) TWS or abs(TWA) out of range (%5.2f, %d)" % (tws, twa))

        twa_vector = self.get_twa_vector(tws)
        bsp_vector = self._get_bsp_vector(tws)
        # print(f"get_bsp, tws: {tws}, twa: {twa}, twa_vector: {str(twa_vector)}, bsp_vector: {str(bsp_vector)}")
        if twa < twa_vector[0]:  # twa is closer to wind than minimum TWA in
            return 0.0
        if twa == round(twa_vector[-1]):
            return bsp_vector[-1]
        if twa > twa_vector[-1]: # twa is euqal or more downwind than max TWA in vector
            factor = (180 - twa) / (180 - twa_vector[-1])
            result = bsp_vector[-1] * min(max(factor, 0.5), 1.0)
            # print(f"twa: {twa}, twa_vector[-1]: {twa_vector[-1]}, factor: {factor}, bsp_vector[-1]: {bsp_vector[-1]}, result: {result}")
            return result
        # find twa interval in TWA
        for i in range(0, len(twa_vector)):
            # if twa == TWA no interpolation required
            if twa == round(twa_vector[i]):
                return bsp_vector[i]
            # interpolate between TWA[i] and TWA[i+1]
            # print(f"twa: {twa}, i: {i}, len(twa_vector): {len(twa_vector)}, twa_vector: {str(twa_vector)}")
            if twa > twa_vector[i] and twa < twa_vector[i + 1]:
                d = twa_vector[i + 1] - twa_vector[i]
                f1 = 1 - (twa - twa_vector[i]) / d
                f2 = 1 - (twa_vector[i + 1] - twa) / d
                return bsp_vector[i] * f1 + bsp_vector[i + 1] * f2
        raise PolarException("no wind value calculated")

# end PolarObject

class Itinerary(object):

    #    def __init__(self, route, start_time):       # constructor with route and start time
    def __init__(self, id, route, start_time):
        self.id = id
        _org = route.wpt(0)
        self.dest = route.wpt(1)
        self.tpt_list = []  # track point list (track point = way point + time)
        tpt = TrackPoint(_org.get_lat(), _org.get_long(), start_time)
        self.tpt_list.append(tpt)
        self.courselist = []

    def get_id(self):
        """ get id of itinerary

        :return: int - itinerary id
        """
        return(self.id)

    def last_tpt(self):
        return self.tpt_list[-1]

    def tpt_count(self):
        return len(self.tpt_list)

    def get_tpt(self, i):
        return self.tpt_list[i]

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

    def elap_time(self):
        """return elapsed time of itinerary

        :return: float - elapsed time of itinerary in milli seconds
        """
        start  = self.tpt_list[0].get_time()
        finish = self.tpt_list[-1].get_time()
        milliseconds = (finish - start) / np.timedelta64(1, 'ms')
        return milliseconds

    def distance(self):
        """return effective distance of itinerary

        :return: float - effective distance of itinerary in nm
        """
        start = self.tpt_list[0].get_wpt()
        finish = self.tpt_list[-1].get_wpt()
        return start.dtw(finish)

    def avg_bsp(self):
        """return average boat speed over itinerary
        :return: float - average boat speed in kn
        """
        return self.distance() / (self.elap_time() / (1000 * 3600))


    def get_courselist(self):
        """ get courselist - list of subsequent courses between rack points

        :return: list of int - list of courses submitted in add_step()
        """
        return self.courselist

    def add_step(self, wo, po, step, crs):
        self.courselist.append(crs)
        _cur_loc = self.last_tpt().get_wpt()
        _cur_time = self.last_tpt().get_time()
        _twd = wo.get_twd(self.last_tpt())
        _tws = wo.get_tws(self.last_tpt())
        if wo.current:
            _set = wo.get_set(self.last_tpt())
            _dft = wo.get_dft(self.last_tpt())
        else:
            _set = 0
            _dft = 0.0
        # print(f"set: {_set} gr, dft: {_dft} nm")
        # print(f"add_step, crs: {crs}, _twd: {_twd}, TWA: {TWA(twd=_twd, hea=crs)} ")
        _bsp = po.get_bsp(tws=_tws, twa=TWA(twd=_twd, hea=crs))
        _dtw = self.dist_to_go()

        if _bsp * step > _dtw:
            duration = _dtw / _bsp
            vector = sum_vectors(crs, _bsp * duration, _set, _dft * duration)
            new_loc = new_crd(lat=_cur_loc.get_lat(), long=_cur_loc.get_long(), hea=vector['dir'], dist=vector['len'])
            self.add_tpt(new_loc['lat'], new_loc['long'], _cur_time + np.timedelta64(int(duration * 3600 * 1000), 'ms'))
        else:
            vector = sum_vectors(crs, _bsp * step, _set, _dft * step)
            new_loc = new_crd(lat=_cur_loc.get_lat(), long=_cur_loc.get_long(), hea=vector['dir'], dist=vector['len'])
            self.add_tpt(new_loc['lat'], new_loc['long'], _cur_time + np.timedelta64(int(step * 3600 * 1000), 'ms'))

# end Itinerary

def TWA(twd, hea):
    """calculates true wind angle for a given wind direction and heading

    :param twd: int - true wind direction in degrees
    :param hea: int - boat heading in degrees
    :return: int - true wind angle between heading and twd, negative is port, positive is starboard
    """
    # print(f"TWA(twd={twd}, hea={hea})")
    r = round(twd - hea)
    # print(f"-TWA-, twd: {twd}, hea: {hea}, r: {r}")
    if r >= -179 and r <= 180:
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
    return degrees(atan2(y, x)) % 360
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

def opposite_direction(dir):
    """ return opposite direction of a course

    :param dir: int direction in degrees
    :return: int - opposite direction in degrees
    """
    return (dir + 180) % 360
# opposite_direction()

def sum_vectors(v1_dir, v1_len, v2_dir, v2_len):
    """add two vectors with direction and length
    """
    vertical   = cos(radians(v1_dir)) * v1_len + cos(radians(v2_dir)) * v2_len
    # print(f"vertical: {vertical}")
    horizontal = sin(radians(v1_dir)) * v1_len + sin(radians(v2_dir)) * v2_len
    # print(f"horizonal: {horizontal}")
    dir = degrees(atan2(horizontal, vertical)) % 360
    len = sqrt(pow(vertical, 2) + pow(horizontal, 2))
    return {'dir': dir, 'len': len}

def version():
    """version function returns version string

    :return: string - version string
    """
    return("Routing library version Oosterschelde 0.1.0 (beta)")


# marks = {"EA1"    :["52° 34,777'", "5° 13,433'"], \
#          "KG10"   :["52° 40,656'", "5° 16,039'"], \
#          "SPT-I"  :["52° 31,998'", "5° 21,390'"], \
#          "SPT-H"  :["52° 31,500'", "5° 7,999'"], \
#          "SPT-G"  :["52° 36,496'", "5° 8,000'"]}

################################ that's all folks ###########################################################
