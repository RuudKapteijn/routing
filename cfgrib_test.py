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
ds = xr.open_dataset('20190823_071554_.grb', engine='cfgrib', backend_kwargs={'filter_by_keys':{'numberOfPoints':609}})
print(ds)
print(ds.u10)
