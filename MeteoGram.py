import Routing as rt
from sys import version
from datetime import date, datetime, timedelta
from math import atan2, pi, degrees, radians, sqrt

# start of script
print(f"Start of MeteoGram.py running on {version} at {datetime.now()}")

mywo = rt.WindObject(windfile='gfs.nc', currentfile='current.nc')
# ds1 = mywo._get_wind_dataset()
# print(ds1)
# ds2 = mywo._get_current_dataset()
# print(ds2)

loc = rt.WayPoint(name="SCH", lat="52° 07,717'", long="004° 14,209'")
tm = datetime.strptime('2020-03-02 02:00', '%Y-%m-%d %H:%M')
mywo.meteogram(location=loc, start=tm, step=timedelta(hours=1), iterations=20)

print(f"Finish at {datetime.now()}")

