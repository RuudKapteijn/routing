##############################################################################################################
# test script for routing library
#
##############################################################################################################
import Routing as rt
import numpy as np
from sys import version
from datetime import datetime
from copy import deepcopy

def ind_fitness(ind):

    # print(f"{ind.dist_to_go()}, {ind.elap_time()}")

    if ind.dist_to_go() < 0.01:             # destination reached
        distance_score = 10000
    else:
        distance_score = int(100 / ind.dist_to_go())

    time_score = int(100000 / ind.elap_time())

    return distance_score + time_score * 5


def list_itineries(il):
    print("list itineries - count: %d" % len(il))

def calculate_minimal_distance(il):
    dist = 100000
    for it in il:
        if it.dist_to_go() < dist:
            dist = it.dist_to_go()
    return dist

def check_dest_reached(il):
    res = False
    for it in il:
        if it.dist_to_go() < 0.01:
            res = True
    return res

def itinery_add_step(il, i, wo, po, step):
    print("add step %d" % i)
    # copy itinerary
    it = deepcopy(il[i])
    _destin = it.get_dest()
    _start = it.last_tpt().get_wpt()
    _btw = _start.btw(_destin)
    _dtw = _start.dtw(_destin)
    _bsp = po.get_bsp(tws=wo.get_tws(it.last_tpt()), twa=rt.TWA(twd=wo.get_twd(it.last_tpt()), hea=_btw))
    print("TWS: %5.2f, TWD: %d, DTW: %5.2f, BTW: %d, TWA: %d, BSP: %5.2f\n" % \
          (wo.get_tws(it.last_tpt()), wo.get_twd(it.last_tpt()), _dtw, _btw, rt.TWA(twd=wo.get_twd(it.last_tpt()), hea=_btw), _bsp))

    if _bsp * step > _dtw:      # destination reached
        duration = _dtw / _bsp  # duration in hours
        new_loc = rt.new_crd(lat=it.last_tpt().get_lat(), long=it.last_tpt().get_long(), hea=_btw, dist=_bsp * duration)
        it.add_tpt(new_loc['lat'], new_loc['long'], it.last_tpt().get_time() + np.timedelta64(int(duration * 60), 'm'))
    else:
        new_loc = rt.new_crd(lat=it.last_tpt().get_lat(), long=it.last_tpt().get_long(), hea=_btw, dist=_bsp * step)
        it.add_tpt(new_loc['lat'], new_loc['long'], it.last_tpt().get_time() + np.timedelta64(int(step * 60), 'm'))
    it.courselist.append(int(_btw))

    il.append(it)               # add new itinerary to list
    del(il[i])                  # remove old itinerary

# start of script
print("Start of RoutingTest running on %s at %s" % (version, datetime.now()))

mywo = rt.WindObject('gfs_new.grb.netcdf')      # open netcdf version of a GRIB file in object mywo
mypo = rt.PolarObject('Mango RaceV0.txt')       # open Expedition polar file in object mypo
org = rt.WayPoint("EA1", "52° 34,777'", "5° 13,433'")
dst = rt.WayPoint("KG10", "52° 40,656'", "5° 16,039'")
rte = rt.Route("myRoute")
rte.add_wpt(org)
rte.add_wpt(dst)
start_time = np.datetime64('2020-01-31 12:00')
print("\n%s at %s" % (rte.stats(), str(start_time)))

try:
    STEPTIME = 0.25          # steps of 15 minutes / quarter hour

    itineraries = []
    it = rt.Itinerary(id="*", route=rte, start_time=start_time)             # create first itinerary
    itineraries.append(it)                                          # put first itinerary in list
    progress = True
    dest_reached = False
    iter = 1
    while progress and not dest_reached:
        print("----------------- iteration %d -----------------------" % iter)
        list_itineries(itineraries)
        minimal_distance_before = calculate_minimal_distance(itineraries)   # minimal dist to dest over all itineraries
        for i in range(0, len(itineraries)):
            itinery_add_step(itineraries, i, mywo, mypo, STEPTIME)
        dest_reached = check_dest_reached(itineraries)                      # check if we are there
        minimal_distance_after = calculate_minimal_distance(itineraries)    # check minimal dist again
        progress = minimal_distance_after < minimal_distance_before         # check if we made progress
        iter += 1

    print("dest_reached: %s" % (str(dest_reached)))
    it = itineraries[0]
    print("Best individual distance: %f, minutes: %f, fitness: %d, courses: %s" %
          (it.dist_to_go(), it.elap_time(), ind_fitness(it), str(it.get_courselist())))

except Exception as e:
    print(e.message)

print("End of script at %s" % datetime.now())