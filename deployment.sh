#!/bin/bash
sudo apt update
sudo apt upgrade
sudo apt-get install python3-pip
pip3 install xarray
pip3 install netCDF4
pip3 install scipy
echo python3 GA_Routing.py ShortRoute.ini
