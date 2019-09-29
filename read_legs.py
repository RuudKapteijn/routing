##################################################################################################################
# read legs from MYsql routing database to Pandas dataframe
#
# Source: MySQL database table legs
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
from datetime import datetime
from sqlalchemy import create_engine
import pymysql
import pandas as pd


print("Start of load_legs at %s" % (str(datetime.now())))

# read legs in dataframe from database routing table legs
sqlEngine = create_engine('mysql+pymysql://root:Mentos2016@34.76.88.166/routing', pool_recycle=3600)
dbConn = sqlEngine.connect()
legs = pd.read_sql("select * from legs", dbConn);
dbConn.close()

pd.set_option('display.expand_frame_repr', False)
print(legs)

print("start mark of row 4: %s" % (legs["start_mark"][4]))

print("End of load_legs at %s" % (str(datetime.now())))
