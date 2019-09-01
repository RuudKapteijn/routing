##################################################################################################################
# Load marks for 24-hour race in routing database
# Source: Excel file from https://www.24uurszeilrace.nl/WedstrijdInfo/Rakken.aspx
# Target: MySQL database table legs
# Required preparation of the Excel file:
# 1. remove superfluous rows 1-2,  17-20, 134-137
# 2. add finish leg from WV19 to FINISH, length 4,9, Max Aantal = 1
#
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
import pymysql
import sys
import pandas as pd
from datetime import datetime
from decimal import Decimal

excelFileName = 'rakken_01-09-2019.xlsx'
sheetName     = 'Boeien'

# start of program
print("Start of load_legs at %s" % (str(datetime.now())))

# read Excel sheet into Pandas dataframe
df = pd.read_excel(excelFileName)
print("%d legs read from Excelsheet" % (len(df.index)))

# create connection to Cloud SQL database RuudsMySQL on public IP-address 34.76.88.166 and open cursor
cnxn = pymysql.connect(host='34.76.88.166', user='root', password='Mentos2016', db='routing')
cursor = cnxn.cursor()
# empty table marks
statement = "DELETE FROM legs"
cursor.execute(statement)

# iterate over dataframe containing legs
count = 0
for index, row in df.iterrows():
    afstand = str(row["Afstand"]).replace(",",".")
    record = (row["Start"], row["Eind"], afstand, row["Max Aantal"])
    # # print(record)
    statement = "INSERT INTO legs(start_mark, dest_mark, distance, max_usage) VALUES(%s, %s, %s, %s)"
    if cursor.execute(statement, record) != 1:
        print("Error occurred, could not insert record %d" % (count + 1))
    count += 1
    if row["Max Aantal"] == 2:
        record = (row["Eind"], row["Start"], afstand, row["Max Aantal"])
        statement = "INSERT INTO legs(start_mark, dest_mark, distance, max_usage) VALUES(%s, %s, %s, %s)"
        if cursor.execute(statement, record) != 1:
            print("Error occurred, could not insert record %d" % (count + 1))
        count += 1

cnxn.commit()
print("%d legs loaded" % (count))
cursor.close()
cnxn.close()
# end of program
print("End of load_legs at %s" % (str(datetime.now())))
