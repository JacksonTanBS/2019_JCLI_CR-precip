#!/usr/bin/python

'''This script provides the supporting functions.'''

import numpy as np
import sys
from calendar import monthrange
from datetime import datetime, timedelta

crpath = ''     # path to cloud regime data
datapath = ''   # path to intermediate processed data


def getMonths(year, year0, month0, year1, month1):
    if   year0 == year1: return range(month0, month1 + 1)
    elif year  == year0: return range(month0, 13)
    elif year  == year1: return range(1, month1 + 1)
    else               : return range(1, 13)


def get_3h_CRs(year, month, sat):

    if sat == 'A':
        CRfile = 'myd_modis_crs_20021201_20170531.gdat'
        UTCdir = 'utc_aqua'
    elif sat == 'T':
        CRfile = 'mod_modis_crs_20021201_20170531.gdat'
        UTCdir = 'utc_terra'
    else:
        sys.exit('Error: sat label undefined.')

    nt, ny, nx = 5296, 180, 360

    t0 = (datetime(year, month, 1) - datetime(2002, 12, 1)).days
    t1 = t0 + monthrange(year, month)[1]

    # read the daily CRs
    CRday = np.fromfile('%s%s' % (crpath, CRfile), 
                        dtype = '<f').reshape(nt, ny, nx)[t0 : t1]

    # get the UTC times
    UTC = np.zeros(CRday.shape)
    tt = 0
    for day in range(1, monthrange(year, month)[1] + 1):
        ts = '%4d%02d%02d' % (year, month, day)
        with open('%s%s/%s.dat' % (crpath, UTCdir, ts), 'r') as f:
            UTC[tt] = np.array([float(line[8 : 13]) if line[:8] == ts else 0 
                                for line in f.readlines()]).reshape(ny, nx)
        tt += 1

    # use the UTC times to get the 3-h CRs
    CR = np.zeros([(t1 - t0) * 8, ny, nx], dtype = 'i4')
    for hr3 in range(8):
        inTime = (UTC > (hr3 * 3)) * (UTC <= (hr3 + 1) * 3)
        CR[hr3 : : 8] = CRday * inTime

    return CR


def get_hhr_CRs(year, month, sat):

    if sat == 'A':
        CRfile = 'myd_modis_crs_20021201_20170531.gdat'
        UTCdir = 'utc_aqua'
    elif sat == 'T':
        CRfile = 'mod_modis_crs_20021201_20170531.gdat'
        UTCdir = 'utc_terra'
    else:
        sys.exit('Error: sat label undefined.')

    nt, ny, nx = 5296, 180, 360

    t0 = (datetime(year, month, 1) - datetime(2002, 12, 1)).days
    t1 = t0 + monthrange(year, month)[1]

    # read the daily CRs
    CRday = np.fromfile('%s%s' % (crpath, CRfile), 
                        dtype = '<f').reshape(nt, ny, nx)[t0 : t1]

    # get the UTC times
    UTC = np.zeros(CRday.shape)
    tt = 0
    for day in range(1, monthrange(year, month)[1] + 1):
        ts = '%4d%02d%02d' % (year, month, day)
        with open('%s%s/%s.dat' % (crpath, UTCdir, ts), 'r') as f:
            UTC[tt] = np.array([float(line[8 : 13]) if line[:8] == ts else 0 
                                for line in f.readlines()]).reshape(ny, nx)
        tt += 1

    # use the UTC times to get the half-hour CRs
    CR = np.zeros([(t1 - t0) * 48, ny, nx], dtype = 'i4')
    for hhr in range(48):
        inTime = (UTC > (hhr * 0.5)) * (UTC <= (hhr + 1) * 0.5)
        CR[hhr : : 48] = CRday * inTime

    return CR


def get_processed_data(var, year0, month0, year1, month1, tres, 
                       sats = ('A', 'T'), regimes = range(1, 13)):

    import gzip
    import pickle

    if   tres == 'hhr':
        npixel = 100
    elif tres == '3hr':
        npixel = 600

    # initialize the array depending on variable
    if   var[:4] == 'rank':
        if 'Num' in var:
            data = {(sat, regime): np.zeros(npixel, 'i8') for sat in sats 
                    for regime in regimes}
        else:
            data = {(sat, regime): np.zeros(npixel, 'f8') for sat in sats 
                    for regime in regimes}
    elif var == 'count':
        data = {(sat, regime): 0 for sat in sats for regime in regimes}
    else:
        data = {(sat, regime): [] for sat in sats for regime in regimes}

    for year in range(year0, year1 + 1):
        for month in getMonths(year, year0, month0, year1, month1):

            with gzip.open('%s%s/%s.%4d%02d.p.gz' % (datapath, tres, var, 
                                                     year, month), 
                           'rb') as f:
                dataMonth = pickle.load(f)

            for sat in sats:
                for regime in regimes:
                    data[sat, regime] += dataMonth[sat, regime]

    return data


def collapse_sat(datain, regimes = range(1, 13)):

    if type(datain['A', regimes[0]]) == np.ndarray:
        dataout = {regime: np.zeros(len(datain['A', regimes[0]]), 
                                    datain['A', regimes[0]].dtype)
                   for regime in regimes}        
    else:
        dataout = {regime: type(datain['A', regimes[0]])() 
                   for regime in regimes}
    
    for regime in regimes:
        dataout[regime] = datain['A', regime] + datain['T', regime]

    return dataout


def plot_scatter_density(ax, x, y, s = 1, n1 = 1000000, n2 = 500, 
                         cmap = 'hot_r'):

    from scipy.stats import gaussian_kde

    # how much to sub-sample so that...
    if n1 == None:
        sub1 = 1
    else:
        sub1 = int(len(x) / n1)    # n1 samples are plotted
    sub2 = int(len(x) / n2)    # n2 pairs are used to compute kernel

    x, y = np.array(x[::sub1]), np.array(y[::sub1])
    xy = np.vstack([x, y])

    z = gaussian_kde(xy[:, ::sub2])(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    # don't plot points with zero density values (white color)
    x = x[z != 0]
    y = y[z != 0]
    z = z[z != 0]

    ax.scatter(x, y, s = s, c = z / z.max(), vmin = 0, vmax = 1,
               cmap = cmap, rasterized = True)

    return None
