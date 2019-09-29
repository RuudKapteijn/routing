##################################################################################################################
# Complement marks for 24-hour race in routing database
#
# Target: MySQL database table legs
# create table legs (start_mark    varchar(20),
# 				     dest_mark     varchar(20),
#                    start_lat     decimal(7,4),
#                    start_lon     decimal(7,4),
#                    dest_lat      decimal(7,4),
#                    dest_lon      decimal(7,4),
#                    distance      decimal(5,2),
#                    max_usage     integer,
#                    distance_calc decimal(5,2),
#                    bearing_calc  decimal(3,0),
#                    valid_from    datetime,
#                    valid_until   datetime,
#                    grib_lat_ind  integer,
#                    grib_lon_ind  integer,
#                    TWD           decimal(3,0),
#                    TWS           decimal(4,2),
#                    TWA           decimal(3,0),
#                    AWA           decimal(3,0),
#                    BSP           decimal(4,2),
#                    cur_dir       decimal(3,0),
#                    cur_spd       decimal(4,2),
#                    SOG           decimal(4,2),
#                    tack          varchar(20),
#                    has_lock      integer,
#                    lock_comp     decimal(4,2),
#                    duration      integer,
#                    sailplan      integer,
#                    primary key   (start_mark, dest_mark))
#
# UPDATE legs SET TWD = 270, TWS = 12.0, valid_from = '2019-08-23 18:40:00', valid_until = '2019-08-24 19:40:00'
#
import pymysql
from datetime import datetime
from decimal import Decimal
from math import sin, cos, sqrt, atan2, radians, degrees, fmod

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

# start of program
print("Start of compl_legs at %s" % (str(datetime.now())))

# create connection to Cloud SQL database RuudsMySQL on public IP-address 34.76.88.166 and open cursor
cnxn = pymysql.connect(host='34.76.88.166', user='root', password='Mentos2016', db='routing')
updcursor =cnxn.cursor()
loopcursor = cnxn.cursor()
statement = "SELECT start_mark, dest_mark, start_lat, start_lon, dest_lat, dest_lon, distance FROM legs"
loopcursor.execute(statement)
rowcount = loopcursor.rowcount
print("%d records found" % (rowcount))

for i in xrange(0,rowcount):
    row = loopcursor.fetchone()
    start_lat = float(row[2])
    start_lon = float(row[3])
    dest_lat  = float(row[4])
    dest_lon  = float(row[5])

    b_cal = bearing(start_lat, start_lon, dest_lat, dest_lon)
    d_cal = distance(start_lat, start_lon, dest_lat, dest_lon)
    print("%8s --> %8s,\t dist: %5.2f, bearing: %3d" % (row[0], row[1], d_cal, b_cal))   # start_mark, dest_mark

    upd_stat = "UPDATE legs SET bearing_calc = " + str(b_cal) + ", distance_calc = " + str(d_cal) + " WHERE start_mark = '" + row[0] +"' AND dest_mark = '" + row[1] +"'"
    print("%s" % (upd_stat))
    updcursor.execute(upd_stat)

cnxn.commit()
loopcursor.close()
updcursor.close()
cnxn.close()
# end of program
print("End of compl_legs at %s" % (str(datetime.now())))
