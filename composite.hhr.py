#!/usr/bin/python

'''This script composites IMERG subgrid precipitation with MODIS CRs and for
each month and saves the output.'''

import numpy as np
import h5py
import gzip
import pickle
import sys
from glob import glob
from calendar import monthrange
from datetime import datetime, timedelta
from func import *


# Define the preliminaries


year, month = int(sys.argv[1]), int(sys.argv[2])

crpath = ''     # path to cloud regime data
imergpath = ''  # path to IMERG data
datapath = ''   # path to intermediate processed data

sats = ('A', 'T')
regimes = range(1, 13)

ntime = monthrange(year, month)[1] * 48
nlon = 360
nlat = 180
lat0, lat1 = 30, 150
npixel = 100


# Get the half-hour CRs


CR = {sat: np.zeros([ntime, nlat, nlon], dtype = 'i4') for sat in sats} 
for sat in sats:
    CR[sat] = get_hhr_CRs(year, month, sat)


# Initialize the arrays.


meanPrecip = {}
numNonzero = {}
numLight = {}
numModer = {}
numHeavy = {}
rankSumPrecip = {}
rankNumNonzero = {}
rankNumLight = {}
rankNumModer = {}
rankNumHeavy = {}
rankAmtLight = {}
rankAmtModer = {}
rankAmtHeavy = {}
count = {}

for sat in sats:
    for regime in regimes:
        meanPrecip[sat, regime] = []
        numNonzero[sat, regime] = []
        numLight[sat, regime] = []
        numModer[sat, regime] = []
        numHeavy[sat, regime] = []
        rankSumPrecip[sat, regime] = np.zeros(npixel, 'f8')
        rankNumNonzero[sat, regime] = np.zeros(npixel, 'i8')
        rankNumLight[sat, regime] = np.zeros(npixel, 'i8')
        rankNumModer[sat, regime] = np.zeros(npixel, 'i8')
        rankNumHeavy[sat, regime] = np.zeros(npixel, 'i8')
        rankAmtLight[sat, regime] = np.zeros(npixel, 'f8')
        rankAmtModer[sat, regime] = np.zeros(npixel, 'f8')
        rankAmtHeavy[sat, regime] = np.zeros(npixel, 'f8')
        count[sat, regime] = 0


# Perform the composite with IMERG for the month.


files = sorted(glob('%s2015/01/01/*.HDF5' % imergpath))
with h5py.File(files[0], 'r') as f:
    fv = f['Grid/precipitationCal'].attrs['_FillValue']

tt = 0

for day in range(1, monthrange(year, month)[1] + 1):

    files = sorted(glob('%s%4d/%02d/%02d/*.HDF5' % (imergpath, year, month, day)))

    for hhr in range(48):

        with h5py.File(files[hhr], 'r') as f:
            P = f['Grid/precipitationCal'][:].T

        for sat in sats:
            for regime in regimes:
                for lat, lon in zip(*np.where(CR[sat][tt] == regime)):

                    p = np.sort(P[lat * 10 : (lat + 1) * 10, 
                                  lon * 10 : (lon + 1) * 10].flatten())
                    if np.all(p != fv) and (lat0 <= lat < lat1):

                        meanPrecip[sat, regime].append(np.mean(p))
                        numNonzero[sat, regime].append(np.sum(p != 0))

                        rankSumPrecip[sat, regime] += p
                        rankNumNonzero[sat, regime] += (p > 0)

                        isLight = (p >= 0.1) * (p < 1)
                        numLight[sat, regime].append(np.sum(isLight))
                        rankNumLight[sat, regime] += isLight
                        rankAmtLight[sat, regime] += p * isLight

                        isModer = (p >= 1) * (p < 10)
                        numModer[sat, regime].append(np.sum(isModer))
                        rankNumModer[sat, regime] += isModer
                        rankAmtModer[sat, regime] += p * isModer

                        isHeavy = (p >= 10)
                        numHeavy[sat, regime].append(np.sum(p >= 10))
                        rankNumHeavy[sat, regime] += isHeavy
                        rankAmtHeavy[sat, regime] += p * isHeavy

                        count[sat, regime] += 1

        tt += 1


# Save the files


variables = {'meanPrecip': meanPrecip,
             'numNonzero': numNonzero,
             'numLight': numLight,
             'numModer': numModer,
             'numHeavy': numHeavy,
             'rankSumPrecip': rankSumPrecip,
             'rankNumNonzero': rankNumNonzero,
             'rankNumLight': rankNumLight,
             'rankNumModer': rankNumModer,
             'rankNumHeavy': rankNumHeavy,
             'rankAmtLight': rankAmtLight,
             'rankAmtModer': rankAmtModer,
             'rankAmtHeavy': rankAmtHeavy,
             'count': count}

for k, v in variables.items():
    with gzip.open('%s%s.%4d%02d.p.gz' % (datapath, k, year, month), 'wb') as f:
        pickle.dump(v, f)
