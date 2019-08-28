#################################################################################
#
# cfgrib test program
# requires installs:
# pip3 install eccodes-python
# pip3 install cfgrib
# apt install libeccodes-dev
# apt install libeccodes-tools
# apt install python3-eccodes
# python3 -m cfgrib selfcheck -> Found ecCodes v2.6.0. Your system is ready
# pip3 install xarray
#
# 20190823_071554_.grb is one of the grib files used during last 24-hrs race
#

import xarray as xr
import math
ds = xr.open_dataset('20190823_071554_.grb', engine='cfgrib', backend_kwargs={'filter_by_keys':{'numberOfPoints':609}})

# print(ds)
# print("---------------------------------------------------")
# print(ds.u10)
# print("---------------------------------------------------")
# print(ds.v10)

# for the upper left corner (lon=1, lat=1) print wind data over time
for i in range(41):
    # http://colaweb.gmu.edu/dev/clim301/lectures/wind/wind-uv
    u10 = ds.u10.values[i,1,1]                           # GRIB message 165, eastward component
    v10 = ds.v10.values[i,1,1]                           # GRIB message 166, northward component
    TWS = math.sqrt(u10 ** 2 + v10 ** 2) * 3600 /1852    # m/s to kn
    TWD = math.atan2(u10, v10) * 180 / math.pi + 180
    print("i={:2d} u10={:5.2f} v10={:5.2f} TWS={:5.2f} TWD={:5.2f}".format(i, u10, v10, TWS, TWD))
