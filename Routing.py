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

# from __future__ import annotations                  # allows forward references to class definitions (e.g. WayPoint)
import numpy as np  # type: ignore
import xarray as xr
from math import sin, asin, cos, sqrt, atan2, radians, degrees, pi
import xml.etree.ElementTree as et
from datetime import datetime, timedelta
from typing import Union, List, Dict

class GribIndexException(Exception):
    """ Exception subclass to throw for index out of range of GRIB file (lat, long or time)

    constructor parameters: message - string - error message
    """
    def __init__(self, message: str):
        self.message:str = message
# GribIndexException

class PolarException(Exception):
    """ Exception subclass to throw for error in polar class

    constructor parameters: message - string - error message
    """
    def __init__(self, message: str):
        self.message:str = message
# PolarException

class WayPoint(object):
    """ Waypoint object 2D position with latitude and longitude

    TODO: add method to return coordinates in format deg min.dec (string)
    """

    def __init__(self, name: str, lat: Union[float, str], long: Union[float, str]) -> None:
        """ Waypoint constructor - creates waypoint

        :param name: string - name of the waypoint
        :param lat:  float  - latitude of the waypoint as degrees.decimal degrees or string - converted with dec_crd()
        :param long: float  - longitude of the waypoint as degrees.decimal degrees or string - converted with dec_crd()
        """
        self.lat: float
        self.long: float
        if isinstance(lat, str):
            self.lat = self.dec_crd(lat)
        if isinstance(lat, float):
            self.lat = lat
        if isinstance(long, str):
            self.long = self.dec_crd(long)
        if isinstance(long, float):
            self.long = long
        self.name: str = name

    def get_name(self) -> str:
        return self.name

    @classmethod
    def dec_crd(cls, s: str) -> float:
        """ decimal_coordinate transforms coordinate string deg min.dec to float

        :param s: coordinate in format "52° 34,777'"
        :return: coordinate as float (degrees.decimal degrees)
        """
        c: List[str] = s.split(" ")
        # print("c[0]: -%s-" % (c[0]))
        c0: str = c[0][0:len(c[0]) - 1]
        # c0 = c[0][0:len(c[0]) - 2]               # changed -1 to -2 for UTF-8 character (2 bytes)
        # print("c0: -%s-" % (c0))
        c1: str = c[1][0:len(c[1]) - 1]
        c1 = c1.replace(",", ".")
        # print("c1: -%s-" % (c1))
        return float(c0) + float(c1) / 60

    def dtw(self, wpt) -> float:
        """distance to waypoint calculates distance from self waypoint to wpt
        # def dtw(self, wpt: WayPoint) -> float:
        :param wpt: waypoint to calculate distance to
        :return: float - distance in nautical miles
        """
        return DTW(self.lat, self.long, wpt.get_lat(), wpt.get_long())

    def btw(self, wpt) -> float:
        """bearing to waypoint calculates bearing from self waypoint to wpt
        # def btw(self, wpt: WayPoint) -> float:

        :param wpt: waypoint to calculate bearing to
        :return: int - bearing in degrees
        """
        return BTW(self.lat, self.long, wpt.get_lat(), wpt.get_long())

    def get_lat(self) -> float:
         return self.lat

    def get_long(self) -> float:
        return self.long
# end WayPoint

class Route(object):                                # Route - series of waypoints

    def __init__(self, name: str, gpxfile: str = None) -> None:
        """ Contructor for Route

        if gpxfile is not supplied after creation the waypoint list is empty
        if gpxfile is supplied this is considered the name of a gpx (XML) file with route data
        route is created from file
        :param name - name of the route
        :param gpxfile - name of gpx file containing route data
        """
        self.name: str = name
        self.wpt_list: List[WayPoint] = []
        self.distance: float = 0.0

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
                # tag contains route name - over write name
                self.name = route.find('gpx:name', ns).text # type: ignore
                # find all rtept's (route points) in route
                for routepoint in route.findall('gpx:rtept', ns):
                    _name: str   = routepoint.find('gpx:name', ns).text    # type: ignore
                    _lat: float  = float(routepoint.attrib['lat'])         # type: ignore
                    _long: float = float(routepoint.attrib['lon'])         # type: ignore
                    wp = WayPoint(name = _name, lat = _lat, long = _long)
                    self.add_wpt(wp)

    def add_wpt(self, wpt: WayPoint) -> None:       # add waypoint to the end of the route
        self.wpt_list.append(wpt)
        if self.wpt_cnt() > 1:
            self.distance += self.wpt_list[-2].dtw(self.wpt_list[-1])

    def wpt(self, i: int) -> WayPoint:              # return i-th waypoint of the waypoint list
        return self.wpt_list[i]

    def dst(self) -> float:                         # distance calculate length of route in nm
        return self.distance

    def wpt_cnt(self) -> int:                       # return number of waypoint in the route
        return len(self.wpt_list)

    def stats(self) -> str:                         # statistics - return string of statistics
        res = "Route %s, %d waypoints, length: %5.2f nm, 1st leg (%s - %s) bearing: %d gr, distance: %5.2f" % \
              (self.name, self.wpt_cnt(), self.dst(), self.wpt(0).get_name(), self.wpt(1).get_name(), \
               self.wpt(0).btw(self.wpt(1)), self.wpt(0).dtw(self.wpt(1)))
        return res
# end Route

class TrackPoint(object):

    def __init__(self, lat: float, long: float, time: datetime) -> None:
        """contructor of trackpoint object

        :param lat: float - latitude of trackpoint as degrees.dec degr
        :param long: float - longitude of trackpoint as degrees.dec degr
        :param time: datetime - time of trackpoint
        """
        self.lat: float = lat
        self.long: float = long
        self.time: datetime = time
        self.wpt: WayPoint = WayPoint(name="tmp", lat=self.lat, long=self.long)

    def get_lat(self) -> float:
        return self.lat

    def get_long(self) -> float:
        return self.long

    def get_wpt(self) -> WayPoint:
        return self.wpt

    def get_time(self) -> datetime:
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

    def __init__(self, windfile: str, currentfile: str = None):
        """ WindObject constuctor create wind object

        :param filename: string - name of NETCDF file containing GRIB data
        """
        print("Load wind data from %s" % windfile)
        self.w_ds = xr.open_dataset(windfile, engine='netcdf4')
        if currentfile:
            print("Load currents from: %s" % currentfile)
            self.c_ds = xr.open_dataset(currentfile, engine='netcdf4')
            self.current: bool = True
        else:
            self.current = False
            print("No current data available")
    #__init__()

    def _get_wind_dataset(self):
        # def _get_wind_dataset(self) -> xr.xarray:
        return self.w_ds

    def _get_current_dataset(self):
        # def _get_current_dataset(self) -> xr.xarray:
        return self.c_ds

    def get_current(self, tpt: TrackPoint) -> Dict[str, float]:
        """ _get_current private method to calculate true wind direction and true wind speed at lat, long & time

        :param tpt: track_point with lat, long & time to calculate wind for
        :return: dict with tws: tws as float, twd: twd as float
        """
        set = 0.0
        dft = 0.0
        if self.current:
            u10c = 2 * self.c_ds['UOGRD_2mbelowsealevel'].interp(time=tpt.get_time(), latitude=tpt.get_lat(),
                                                             longitude=tpt.get_long()).values
            v10c = 2 * self.c_ds['VOGRD_2mbelowsealevel'].interp(time=tpt.get_time(), latitude=tpt.get_lat(),
                                                             longitude=tpt.get_long()).values
            if not np.isnan(u10c) and not np.isnan(v10c):
                set = atan2(u10c, v10c) % (pi * 2)
                dft = sqrt(u10c ** 2 + v10c ** 2) * 3600 / 1852

        return {'set': set, 'dft': dft}
    # get_current()

    def get_wind(self, tpt: TrackPoint) -> Dict[str, float]:
        """ return twd and tws for a trackpoint location and time

        :param tpt:  position in lat and long, time in UTC
        :return: twd in radians, tws in knots
        """
        u10w = self.w_ds['UGRD_10maboveground'].interp(time=tpt.get_time(), latitude=tpt.get_lat(),
                                                       longitude=tpt.get_long()).values
        v10w = self.w_ds['VGRD_10maboveground'].interp(time=tpt.get_time(), latitude=tpt.get_lat(),
                                                       longitude=tpt.get_long()).values
        gwd = atan2(u10w, v10w) + pi
        gws = sqrt(u10w ** 2 + v10w ** 2) * 3600 / 1852

        set = 0.0
        dft = 0.0
        if self.current:
            # print(f"read the current at {tpt.get_time()}, {tpt.get_lat()}, {tpt.get_long()}")
            u10c = 2 * self.c_ds['UOGRD_2mbelowsealevel'].interp(time=tpt.get_time(), latitude=tpt.get_lat(),
                                                             longitude=tpt.get_long()).values
            v10c = 2 *self.c_ds['VOGRD_2mbelowsealevel'].interp(time=tpt.get_time(), latitude=tpt.get_lat(),
                                                             longitude=tpt.get_long()).values
            # print(f"u10c: {u10c}, v10c: {v10c}")
            if not np.isnan(u10c) and not np.isnan(v10c):
                # print("we've got something")
                u10w -= u10c
                v10w -= v10c
                set = atan2(u10c, v10c) % (2 * pi)
                dft = sqrt(u10c ** 2 + v10c ** 2) * 3600 / 1852
            else:
                # print("NaN")
                set = 0.0
                dft = 0.0

        twd = atan2(u10w, v10w) + pi
        tws = sqrt(u10w ** 2 + v10w ** 2) * 3600 / 1852

        return {'twd': twd, 'tws': tws, 'gwd': gwd, 'gws': gws, 'set': set, 'dft': dft}
    # get_wind()

    def get_current(self) -> bool:
        return self.current

    def meteogram(self, location: WayPoint, start: datetime, step: timedelta, iterations: int) -> None:
        tm: datetime = start
        print("sp date/time (UTC)      gwd  gws    twd  tws    set   dft")
        for i in range(iterations):
            tp = TrackPoint(location.get_lat(), location.get_long(), tm)
            wind = self.get_wind(tpt=tp)
            print(f"{i+1:2d} {tm}  {degrees(wind['gwd']):003.0f}  {wind['gws']:5.2f}  {degrees(wind['twd']):003.0f}  {wind['tws']:5.2f}  {degrees(wind['set']):003.0f}  {wind['dft']:5.2f}")
            tm += step
    # meteogram()

    @classmethod
    def TWA(cls, twd: float, hea: float) -> float:
        """ Calculate True Wind Angle - Angle between wind direction and heading of the boat

        :param twd: radians
        :param hea: radinas
        :return: degrees
        """
        res = ((twd - hea) + pi) % (2 * pi) - pi
        if res == -pi:
            res = pi
        return degrees(res)
    # TWA()

    @classmethod
    def dt64_to_dt(cls, dt64: np.datetime64) -> datetime:
        """ transform numpy.datetime64 to datetime

        :param dt64:
        :return:
        """
        ts = (dt64 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
        return datetime.utcfromtimestamp(ts)

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
    def __init__(self, id: str, route: Route, start_time: datetime) -> None:
        self.id = id
        _org = route.wpt(0)
        self.dest = route.wpt(1)
        self.leg_distance = route.wpt(0).dtw(route.wpt(1))
        self.tpt_list: List[TrackPoint] = []  # track point list (track point = way point + time)
        tpt = TrackPoint(_org.get_lat(), _org.get_long(), start_time)
        self.tpt_list.append(tpt)
        self.courselist: List[float] = []

    def get_id(self) -> str:
        return self.id

    def last_tpt(self) -> TrackPoint:
        return self.tpt_list[-1]

    def tpt_count(self) -> int:
        return len(self.tpt_list)

    def get_tpt(self, i) -> TrackPoint:
        return self.tpt_list[i]

    def add_tpt(self, lat: float, long: float, time: datetime):
        tpt = TrackPoint(lat, long, time)
        self.tpt_list.append(tpt)

    def get_dest(self) -> WayPoint:
        return self.dest

    def dist_to_go(self) -> float:
        lat1 = self.last_tpt().get_lat()
        long1 = self.last_tpt().get_long()
        lat2 = self.dest.get_lat()
        long2 = self.dest.get_long()
        return DTW(lat1, long1, lat2, long2)

    def elap_time(self) -> float:
        """return elapsed time of itinerary

        :return: float - elapsed time of itinerary in milli seconds
        """
        start  = self.tpt_list[0].get_time()
        finish = self.tpt_list[-1].get_time()
        seconds = (finish - start) / timedelta(seconds=1)
        return seconds

    def distance(self) -> float:                    # return effective distance of itinerary in nm
        start = self.tpt_list[0].get_wpt()
        finish = self.tpt_list[-1].get_wpt()
        return start.dtw(finish)

    def avg_bsp(self) -> float:                     # return average boat speed over itinerary in kn
        return (self.leg_distance - self.dist_to_go()) / (self.elap_time() / 3600)

    def get_courselist(self) -> List[float]:        # get courselist - list of subsequent courses in radians
        return self.courselist

    def add_step(self, wo, po, step, crs):
        if crs < 0 or crs >= (2 * pi):
            raise ValueError
        self.courselist.append(crs)
        _cur_loc = self.last_tpt().get_wpt()
        _cur_time = self.last_tpt().get_time()
        wind: Dict[str, float] = wo.get_wind(tpt=self.last_tpt())
        _twd = wind['twd']
        _tws = wind['tws']
        _set = wind['set']
        _dft = wind['dft']
        # print(f"set: {_set} gr, dft: {_dft} nm")
        # print(f"add_step, crs: {crs}, _twd: {_twd}, TWA: {TWA(twd=_twd, hea=crs)} ")
        _bsp = po.get_bsp(tws=_tws, twa=WindObject.TWA(twd=_twd, hea=crs))
        _dtw = self.dist_to_go()

        if _bsp * step > _dtw:
            duration = _dtw / _bsp
        else:
            duration = step

        vector = sum_vectors(crs, _bsp * duration, _set, _dft * duration)
        new_loc = new_crd(lat=_cur_loc.get_lat(), long=_cur_loc.get_long(), hea=vector['dir'], dist=vector['len'])
        self.add_tpt(new_loc['lat'], new_loc['long'], _cur_time + timedelta(hours=duration))

# end Itinerary

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
    _hea = hea
    _lat2 = asin(sin(_lat) * cos(dist / R) + cos(_lat) * sin(dist / R) * cos(_hea))
    _long2 = _long + atan2(sin(_hea) * sin(dist / R) * cos(_lat), cos(dist / R) - sin(_lat) * sin(_lat2))
    return {'lat': degrees(_lat2), 'long': degrees(_long2)}

# def opposite_direction(dir):                # return opposite direction of a course
#     return (dir + 180) % 360
# # opposite_direction()

def sum_vectors(v1_dir, v1_len, v2_dir, v2_len):
    """add two vectors with direction and length
    """
    y = cos(v1_dir) * v1_len + cos(v2_dir) * v2_len
    # print(f"vertical: {vertical}")
    x = sin(v1_dir) * v1_len + sin(v2_dir) * v2_len
    # print(f"horizonal: {horizontal}")
    dir = atan2(x, y) % (2 * pi)
    len = sqrt(pow(x, 2) + pow(y, 2))
    # print(f"sum_vectors(v1_dir={degrees(v1_dir):003.0f}, v1_len={v1_len:5.2f}, v2_dir={degrees(v2_dir):003.0f}, v2_len={v2_len:5.2f} -> x: {x:5.2f}, y: {y:5.2f} dir: {degrees(dir):003.0f}, len: {len:5.2f}")
    return {'dir': dir, 'len': len}

def version():
    """version function returns version string

    :return: string - version string
    """
    return("Routing library version Westerschelde 0.1.1 (beta)")


# marks = {"EA1"    :["52° 34,777'", "5° 13,433'"], \
#          "KG10"   :["52° 40,656'", "5° 16,039'"], \
#          "SPT-I"  :["52° 31,998'", "5° 21,390'"], \
#          "SPT-H"  :["52° 31,500'", "5° 7,999'"], \
#          "SPT-G"  :["52° 36,496'", "5° 8,000'"]}

################################ that's all folks ###########################################################
