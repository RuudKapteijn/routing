# routing
Naval Routing Program

Routing.py is a library containing Python navigation objects
Classes and methods are documented of Routing.py are documented with Doc Strings (e.g. help(Routing.Waypoint)

GA routing implements a Genetic Routing Algoritm
7 March 2020: current implemented
Working on performance.... (currently only uses one core)

requires Python3.6, and packages xarray, netCDF4, numpy and scipy
IMPORTANT: unzip current.zip to current.nc

Azure Virtual Server: Standard A2m v2 (2 vcpus, 16 GiB memory)
Preparation script:

#!/bin/bash
sudo apt update
sudo apt upgrade
sudo apt-get install python3-pip
pip3 install xarray
pip3 install netCDF4
pip3 install scipy
python3 GA_Routing.py ShortRoute.ini




