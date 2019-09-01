##################################################################################################################
# Load marks for 24-hour race in routing database
# Source: Excel file from https://www.24uurszeilrace.nl/WedstrijdInfo/Boeien.aspx
# Target: MySQL database table marks
# create table marks(name varchar(15),
#                    description varchar(30),
#                    type varchar(20),
#                    lat_min_sec varchar(20),
#                    lon_min_sec varchar(20),
#                    lat_min varchar(20),
#                    lon_min varchar(20),
#                    lat_dec decimal(7,4),
#                    lon_dec decimal(7,4),
#                    primary key (name))
#
import pymysql
import sys
import pandas as pd
from datetime import datetime
from decimal import Decimal

def dec_coord(s):
    c = s.split(" ")
    c0 = c[0][0:len(c[0])-1]
    c1 = c[1][0:len(c[1])-1]
    c1 = c1.replace(",",".")
    return Decimal(c0) + Decimal(c1) / 60

excelFileName = 'boeien_01-09-2019.xlsx'
sheetName     = 'Boeien'

# start of program
print("Start of load_marks at %s" % (str(datetime.now())))

# read Excel sheet into Pandas dataframe
df = pd.read_excel(excelFileName)
print("%d marks read from Excelsheet" % (len(df.index)))

# create connection to Cloud SQL database RuudsMySQL on public IP-address 34.76.88.166 and open cursor
cnxn = pymysql.connect(host='34.76.88.166', user='root', password='Mentos2016', db='routing')
cursor = cnxn.cursor()
# empty table marks
statement = "DELETE FROM marks"
cursor.execute(statement)

# iterate over dataframe containing marks
count = 0
for index, row in df.iterrows():
    record = (row["Naam"], row["Omschrijving"], row["Type"], row["Lat (min, sec)"], row["Long (min, sec)"], \
              row["Lat (min)"], row["Long (min)"], dec_coord(row["Lat (min)"]), dec_coord(row["Long (min)"]))
    # print(record)
    statement = "INSERT INTO marks VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s)"
    if cursor.execute(statement, record) != 1:
        print("Error occurred, could not insert record %d" % (count + 1))
    count += 1

cnxn.commit()
print("%d marks loaded" % (count))
cursor.close()
cnxn.close()
# end of program
print("End of load_marks at %s" % (str(datetime.now())))
