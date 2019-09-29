# -*- coding: utf-8 -*-
##################################################################################################################
# Test distance and bearing calculation with four legs in the four quadrants
#
#   NE: EA1 --> KG10        014 6.05
#   SE: EA1 --> SPT-I     119 5.62
#   SW: EA1 --> SPT-H     223 4.66
#   NW: EA1 --> SPT-G     297 3.68
#   SS: SPT-G --> SPT-H 179 4.99
#
from datetime import datetime
from decimal import Decimal
from math import sin, cos, sqrt, atan2, radians, degrees

marks = {"EA1"      :["52° 34,777'", "5° 13,433'"], \
         "KG10"     :["52° 40,656'", "5° 16,039'"], \
         "SPT-I"  :["52° 31,998'", "5° 21,390'"], \
         "SPT-H"  :["52° 31,500'", "5° 7,999'"], \
         "SPT-G"  :["52° 36,496'", "5° 8,000'"]}

def distance(lat1, lon1, lat2, lon2):
    R = 6373 / 1.852
    # print("%f, %f, %f, %f" % (lat1, lon1, lat2, lon2))
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def bearing(lat1, lon1, lat2, lon2):
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    y = sin(dlon) * cos(lat2)
    x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos (dlon)
    return (degrees(atan2(y,x)) + 360) % 360  # fmod((degrees(atan2(y, x)) + 360.0), 360.0)

def dec_coord(s):
    c = s.split(" ")
    # print("c[0]: -%s-" % (c[0]))
    # c0 = c[0][0:len(c[0]) - 1]
    c0 = c[0][0:len(c[0]) - 2]               # changed -1 to -2 for UTF-8 character (2 bytes)
    # print("c0: -%s-" % (c0))
    c1 = c[1][0:len(c[1])-1]
    c1 = c1.replace(",",".")
    return Decimal(c0) + Decimal(c1) / 60

def test(org, dest, d_est, b_est):
    lat_org = dec_coord(marks[org][0])
    lon_org = dec_coord(marks[org][1])
    lat_dst = dec_coord(marks[dest][0])
    lon_dst = dec_coord(marks[dest][1])
    d_cal = distance(lat_org, lon_org, lat_dst, lon_dst)
    b_cal = bearing(lat_org, lon_org, lat_dst, lon_dst)
    print("%s --> %s\td(est): %1.2f, d(cal): %1.2f, b(est): %3d, b(cal): %3d" % (org, dest, d_est, d_cal, b_est, b_cal))

print("Start of dist_bearing_test at %s" % (str(datetime.now())))

test("EA1", "KG10", 6.05, 14)
test("EA1", "SPT-I", 5.62, 119)
test("EA1", "SPT-H", 4.66, 223)
test("EA1", "SPT-G", 3.68, 279)
test("SPT-G", "SPT-H", 4.99, 179)

print("End of dist_bearing_test at %s" % (str(datetime.now())))
