#!/usr/bin/python

'''This script plots the figures for the paper.'''

import numpy as np
import matplotlib.pyplot as plt
from calendar import monthrange
from datetime import datetime, timedelta
from func import *

options = sys.argv[1:]
tres = 'hhr'

year0, year1 = 2014, 2017
month0, month1 = 4, 3
ntime = ((datetime(year1, month1, monthrange(year1, month1)[1]) - 
          datetime(year0, month0, 1)).days + 1) * 8

crpath = ''     # path to cloud regime data
datapath = ''   # path to intermediate processed data

# satellites and regime properties/ranges
sats = ('T', 'A')
satlabel = {'A': 'Aqua', 'T': 'Terra'}
regimes = range(1, 13)
regimes1 = range(1, 7)
regimes2 = range(7, 13)
if   tres == 'hhr':
    npixel = 100
elif tres == '3hr':
    npixel = 600
treslabel = {'hhr': '½-h', '3hr': '3-h'}

# plot configurations
scol = 3.503    # single column (89 mm)
dcol = 7.204    # double column (183 mm)
flpg = 9.724    # full page length (247 mm)
plt.rcParams['figure.figsize'] = (scol, 0.75 * scol)
plt.rcParams['font.size'] = 9
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['axes.titlesize'] = 'medium'
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['font.sans-serif'] = ['TeX Gyre Heros', 'Helvetica',
                                   'DejaVu Sans']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['image.cmap'] = 'viridis_r'
subplttxt = ('(a)', '(b)', '(c)', '(d)', '(e)', '(f)')


#--- CR figures ---#


if 'cr.centroids' in options:

    from matplotlib.colors import BoundaryNorm

    CTPbins = np.array((0, 180, 310, 440, 560, 680, 800, 1100))
    COTbins = np.array(('0', '1.3', '3.6', '9.4', '23', '60', '150'))
    x = np.arange(len(COTbins))
    y = np.arange(len(CTPbins))
    bounds = (0, 0.2, 1, 2, 3, 4, 6, 8, 10, 15, 20, 50)

    centroids = np.loadtxt('%sMODIS.CR12.C6.dat' % (crpath,)).reshape(12, 7, 6)

    fig = plt.figure(figsize = (dcol, 0.6 * dcol))
    plt.subplots_adjust(bottom = 0.15, hspace = 0.15, wspace = 0.15)
    for rr, regime in enumerate(regimes):
        ax = plt.subplot(3, 4, 1 + rr, aspect = 0.6)
        ax.set_title('CR{0:d} (CF = {1:4.1f}%)'.format(regime, 
                     centroids[rr].sum() * 100), loc = 'left', pad = 2)
        mp = ax.pcolormesh(x, y, centroids[rr] * 100, 
                           norm = BoundaryNorm(bounds, 255), 
                           cmap = plt.cm.hot_r)
        ax.set_ylim(ax.set_ylim()[::-1])
        if rr >= 8:
            ax.set_xticks(x)
            ax.set_xticklabels(COTbins, rotation = 45)
        else:
            ax.set_xticks(x)
            ax.tick_params(axis = 'x', bottom = False, labelbottom = False)
        if not rr % 4:
            ax.set_yticks(y)
            ax.set_yticklabels(CTPbins)
        else:
            ax.set_yticks(y)
            ax.tick_params(axis = 'y', left = False, labelleft = False)

    cb = plt.colorbar(cax = plt.axes([0.125, 0.0, 0.775, 0.02]), 
                      mappable = mp, orientation = 'horizontal',
                      ticks = bounds, format = '%g')
    cb.set_label('(%)')
    plt.figtext(0.5, 0.07, 'cloud optical thickness', ha = 'center', va = 'top')
    plt.figtext(0.05, 0.5, 'cloud top pressure (hPa)', va = 'center', 
                rotation = 90)

    fig.savefig('fig.cr.centroids.pdf')
    plt.close()


if 'cr.geogdist' in options:

    from matplotlib.colors import BoundaryNorm
    from mpl_toolkits.basemap import Basemap

    lons, lats = np.meshgrid(np.linspace(-180, 180, 361),
                             np.linspace(-90, 90, 181))

    # read the daily CRs
    nt, ny, nx = 5296, 180, 360
    t0 = (datetime(year0, month0, 1) - datetime(2002, 12, 1)).days
    t1 = (datetime(year1, month1, monthrange(year1, month1)[1]) - 
          datetime(2002, 12, 1)).days + 1
    CR = {}
    CR['A'] = np.fromfile('%smyd_modis_crs_20021201_20170531.gdat' % crpath, 
                          dtype = '<f').reshape(nt, ny, nx)[t0 : t1]
    CR['T'] = np.fromfile('%smod_modis_crs_20021201_20170531.gdat' % crpath, 
                          dtype = '<f').reshape(nt, ny, nx)[t0 : t1]

    # set up a colormap with out-of-range colors
    cmap = plt.cm.magma_r
    cmap.set_under('w')
    cmap.set_over('k')    # does not appear to be working???

    bounds = (1, 2, 3, 4, 6, 8, 10, 13, 16, 20, 25, 30, 40, 50, 70)
    m = Basemap(projection = 'moll', lon_0 = 0, resolution = 'c')

    fig = plt.figure(figsize = (dcol, 0.7 * dcol))
    for rr, regime in enumerate(regimes):
        ax = plt.subplot(4, 3, 1 + rr, aspect = 0.6)
        ax.set_title('CR%d' % regime, loc = 'left', pad = -5)

        rfo = ((np.sum(CR['A'] == regime, 0) + np.sum(CR['T'] == regime, 0)) / 
               (np.sum(CR['A'] != 0, 0) + np.sum(CR['T'] != 0, 0) + 1e-9))
        mp = m.pcolormesh(lons, lats, np.ma.masked_less(rfo * 100, 1), 
                          norm = BoundaryNorm(bounds, 255),
                          cmap = cmap, latlon = True,
                          rasterized = True)
        m.drawcoastlines(color = '0.5', linewidth = 0.5)
        m.drawparallels(np.arange(-60, 61, 30), color = '0.5', 
                        linewidth = 0.3, dashes = (2, 2))
        m.drawmeridians(np.arange(-180, 181, 60), color = '0.5', 
                        linewidth = 0.3, dashes = (2, 2))

    cb = plt.colorbar(cax = plt.axes([0.125, 0.05, 0.775, 0.015]), 
                      mappable = mp, orientation = 'horizontal',
                      ticks = bounds, extend = 'both')
    cb.set_label('(%)')

    fig.savefig('fig.cr.geogdist.pdf')
    plt.close()

    with open('tab.rfo.lat60.txt', 'w') as f:

        for rr, regime in enumerate(regimes):

            rfo = ((np.sum(CR['A'][:, 30 : 150] == regime) + 
                    np.sum(CR['T'][:, 30 : 150] == regime)) / 
                   (np.sum(CR['A'][:, 30 : 150] != 0) + 
                    np.sum(CR['T'][:, 30 : 150] != 0))) * 100

            f.write('CR%d: %5.2f\n' % (regime, rfo))


if 'cr.composite' in options:

    import h5py
    from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
    from mpl_toolkits.basemap import Basemap
    from matplotlib.colorbar import ColorbarBase

    year, month, day = 2015, 1, 1
    t0, t1 = 18, 23
    modpath = '/media/Depot/MODIS/MOD.EqualArea/hhr/'

    # read the UTC data derived from mean SZA
    ts = '%4d%02d%02d' % (year, month, day)
    with open('%sutc_terra/%s.dat' % (crpath, ts), 'r') as f:
        UTC = np.array([float(line[8 : 13]) if line[:8] == ts else 0 
                        for line in f.readlines()]).reshape(180, 360)

    # read the equal-area bounds and grids
    file = ('%s%d/%03d/MOD.EqualArea.v1-0.%d%03d.0000.hdf5' % 
            (modpath, year, day, year, day))
    with h5py.File(file, 'r') as f:
        lat_bounds = f['lat_bounds'][:]
        lon_bounds = f['lon_bounds'][:]
    eqlons, eqlats = np.meshgrid(np.linspace(0.5, 359.5, 360), 
                                 np.linspace(-89.5, 89.5, 180))

    # read the equal-area half-hour data
    numPixel = np.zeros([t1 - t0 + 1, 41252], 'i4')
    for hh, hhr in enumerate(range(t0, t1)):
        file = ('%s%d/%03d/MOD.EqualArea.v1-0.%d%03d.%02d%02d.hdf5' % 
                (modpath, year, day, year, day, hhr // 2, hhr % 2 * 30))
        with h5py.File(file, 'r') as f:
            numPixel[hh] = f['numPixel06'][:]

    # interpolate the equal-area data to equal-angle
    numPixelIntp = np.zeros([t1 - t0 + 1, 180, 360], 'i4')
    for cell in range(41252):
        inCell = ((eqlats >= lat_bounds[cell, 0]) * 
                  (eqlats <  lat_bounds[cell, 1]) * 
                  (eqlons >= lon_bounds[cell, 0]) * 
                  (eqlons <  lon_bounds[cell, 1]))
        for hh in range(t1 - t0 + 1):
            numPixelIntp[hh, inCell] += numPixel[hh, cell]
    numPixelIntp = np.concatenate([numPixelIntp[:, :, 180:], 
                                   numPixelIntp[:, :, :180]], 2)

    # define the equal-angle grid
    lons, lats = np.meshgrid(np.linspace(-179.5, 179.5, 360), 
                             np.linspace(-89.5, 89.5, 180))
    lonedges, latedges = np.meshgrid(np.linspace(-180, 180, 361), 
                                     np.linspace(-90, 90, 181))

    # set up the plot
    m = Basemap(projection = 'moll', lon_0 = 0, resolution = 'c')
    bounds = np.arange(t0 / 2, t1 / 2 + 0.6, 0.5)
    cmap1 = ListedColormap(('#bbd6e8', '#ffd8b6', '#bfe2bf', 
                            '#f2bebe', '#ded1eb', '#dcccc9'))
    cmap1.set_under('w')
    cmap1.set_over('w')
    cols = ('C0', 'C1', 'C2', 'C3', 'C4', 'C5')
    cmap2 = ListedColormap(cols)

    plt.figure(figsize = (scol, 0.7 * scol))
    mp = m.pcolormesh(lonedges, latedges, UTC, 
                      norm = BoundaryNorm(bounds, cmap1.N), cmap = cmap1, 
                      latlon = True, rasterized = True)
    for hhr, col in zip(range(t1 - t0 + 1), cols):
        m.contour(lons, lats, numPixelIntp[hhr] > 0, 1, colors = col, 
                  linewidths = 1, latlon = True)
    m.drawcoastlines(linewidth = 0.5)
    m.drawparallels((-60, 60))
    cb = ColorbarBase(plt.axes([0.125, 0.1, 0.775, 0.025]), cmap = cmap2,
                      orientation = 'horizontal',
                      norm = BoundaryNorm(bounds, cmap2.N))
    cb.set_label('UTC hour')

    plt.savefig('fig.cr.composite.pdf')
    plt.close()


#--- Distribution figures ---#


if 'dist.meanprecip' in options:

    meanPrecip = collapse_sat(get_processed_data('meanPrecip', year0, month0, 
                                                 year1, month1, tres))
    numNonzero = collapse_sat(get_processed_data('numNonzero', year0, month0, 
                                                 year1, month1, tres))

    precipBins = np.logspace(-2, 1, 52)
    xP = precipBins[1:] - 0.5 * np.diff(precipBins)

    fig, ax = plt.subplots(2, 1, sharex = True, sharey = True)
    plt.subplots_adjust(left = 0.15)
    for r1, regimeset in enumerate((regimes1, regimes2)):
        for r2, regime in enumerate(regimeset):
            c, _ = np.histogram(meanPrecip[regime], precipBins)
            ax[r1].plot(xP, c / len(meanPrecip[regime]), lw = 1, 
                        label = 'CR%d' % regime)
            ax[r1].axvline(np.mean(meanPrecip[regime]), color = 'C%d' % r2,
                           lw = 1, ls = '--')
        ax[r1].set_xscale('log')
        ax[r1].legend(ncol = 2, loc = 2, borderpad = 0.2, labelspacing = 0.1,
                      columnspacing = 0.5)
        ax[r1].grid()

    plt.figtext(0.0, 0.5, 'distribution', va = 'center', rotation = 90)
    ax[0].tick_params(axis = 'x', which = 'both', bottom = False, 
                      labelbottom = False)
    ax[1].set_xlabel('mean precip. rate (mm / h)')
    ax[0].set_ylim([-0.002, 0.042])
    fig.savefig('fig.dist.meanprecip.pdf')
    plt.close()


if 'dist.precipfrac' in options:

    numNonzero = collapse_sat(get_processed_data('numNonzero', year0, month0, 
                                                 year1, month1, tres))

    fracBins = np.linspace(0, 1, 26)[1 : -1]
    xF = fracBins[1:] - 0.5 * np.diff(fracBins)

    fig, ax = plt.subplots(2, 1, sharex = True, sharey = True)
    plt.subplots_adjust(left = 0.15)
    for r1, regimeset in enumerate((regimes1, regimes2)):
        for r2, regime in enumerate(regimeset):
            c, _ = np.histogram(np.array(numNonzero[regime]) / npixel, fracBins)
            ax[r1].plot(xF, c / len(numNonzero[regime]), lw = 1,
                        label = 'CR%d' % regime)
            ax[r1].axvline(np.mean(numNonzero[regime]) / npixel, 
                           color = 'C%d' % r2, lw = 1, ls = '--')
        ax[r1].legend(loc = 1, ncol = 2, borderpad = 0.3, labelspacing = 0.2,
                      columnspacing = 1)
        ax[r1].grid()
        ax[r1].set_xlim([-0.05, 1.05])
        if r1 == 0:
            ax[r1].tick_params(axis = 'x', bottom = False, labelbottom = False)
        if r1 == 1:
            ax[r1].set_xlabel('precip. frac.')
    plt.figtext(0.0, 0.5, 'distribution', va = 'center', rotation = 90)
    fig.savefig('fig.dist.precipfrac.pdf')
    plt.close()


if 'dist.stats' in options:

    meanPrecip = collapse_sat(get_processed_data('meanPrecip', year0, month0, 
                                                 year1, month1, tres))
    numNonzero = collapse_sat(get_processed_data('numNonzero', year0, month0, 
                                                 year1, month1, tres))

    with open('tab.dist.meanprecip.txt', 'w') as f:
        for rr, regime in enumerate(regimes):
            f.write('CR%d: %6.4f\n' % (regime, np.mean(meanPrecip[regime])))

    with open('tab.dist.avgfrac.txt', 'w') as f:
        for rr, regime in enumerate(regimes):
            f.write('CR%d: %6.4f\n' % (regime, 
                                    np.mean(numNonzero[regime]) / npixel))

    totalPrecip = sum([sum(meanPrecip[regime]) for regime in regimes])

    with open('tab.dist.contribution.txt', 'w') as f:
        for rr, regime in enumerate(regimes):
            contribution = np.sum(meanPrecip[regime]) / totalPrecip
            f.write('CR%d: %6.4f\n' % (regime, contribution))


#--- Fraction-precipitation figures ---#


if 'frac-precip.density' in options:

    from matplotlib.colorbar import ColorbarBase
    from matplotlib.colors import Normalize

    meanPrecip = collapse_sat(get_processed_data('meanPrecip', year0, month0, 
                                                 year1, month1, tres))
    numNonzero = collapse_sat(get_processed_data('numNonzero', year0, month0, 
                                                 year1, month1, tres))

    fig = plt.figure(figsize = (dcol, 0.75 * scol))
    plt.subplots_adjust(bottom = 0.2)

    for rr, regime in enumerate(regimes):

        ax = fig.add_subplot(2, 6, 1 + rr)
        plot_scatter_density(ax, np.array(numNonzero[regime]) / npixel, 
                             meanPrecip[regime], s = 0.3, n1 = None, 
                             cmap = plt.cm.magma_r)
        ax.text(0, 5, 'CR%d' % regime, va = 'top', ha = 'left')
        ax.set_xticks(np.linspace(0, 1, 3))
        ax.set_yticks(np.linspace(0, 5, 3))
        if rr < 6:
            ax.tick_params(axis = 'x', bottom = False, labelbottom = False)
        if rr % 6:
            ax.tick_params(axis = 'y', left = False, labelleft = False)
        ax.set_ylim([-0.25, 5.25])
        ax.grid()

        # compute the quartile lines
        P = [[] for _ in range(101)]
        for x, y in zip(numNonzero[regime], meanPrecip[regime]):
            P[int(round(x / npixel * 100))].append(y)
        ax.plot(np.linspace(0, 1, 101), 
                [np.percentile(p, 25) for p in P], c = 'C0', lw = 0.25)
        ax.plot(np.linspace(0, 1, 101), 
                [np.percentile(p, 50) for p in P], c = 'C0', lw = 0.50)
        ax.plot(np.linspace(0, 1, 101), 
                [np.percentile(p, 75) for p in P], c = 'C0', lw = 0.25)
        ax.plot(np.linspace(0, 1, 101), 
                [np.mean(p) for p in P], c = 'C0', ls = ':', lw = 0.75)

    cb = ColorbarBase(plt.axes([0.125, 0.0, 0.775, 0.025]),
                      cmap = plt.cm.magma_r,
                      orientation = 'horizontal',
                      norm = Normalize(vmin = 0, vmax = 1))
    cb.set_label('normalized density')

    plt.figtext(0.5, 0.1, 'precip. frac.', ha = 'center', 
                va = 'top')
    plt.figtext(0.06, 0.5, 'mean precip. rate (mm / h)', va = 'center',
                rotation = 90)

    fig.savefig('fig.frac-precip.density.pdf')
    plt.close()


#--- Ranked property figures ---#


if 'ranked' in options:

    rankSumPrecip = collapse_sat(get_processed_data('rankSumPrecip', year0, 
                                 month0, year1, month1, tres))
    rankNumNonzero = collapse_sat(get_processed_data('rankNumNonzero', year0, 
                                  month0, year1, month1, tres))
    count = collapse_sat(get_processed_data('count', year0, month0, 
                                            year1, month1, tres))

    fig = plt.figure(figsize = (dcol, 0.75 * scol))
    plt.subplots_adjust(wspace = 0.25)
    ax = [[] for _ in range(4)]
    ax[0] = plt.subplot(221)
    ax[1] = plt.subplot(223, sharey = ax[0])
    ax[2] = plt.subplot(222)
    ax[3] = plt.subplot(224, sharey = ax[2])

    for r1, regimeset in enumerate((regimes1, regimes2)):
        for r2, regime in enumerate(regimeset):
            mean = rankSumPrecip[regime] / count[regime]
            ax[r1].plot(range(1, npixel + 1), mean, color = 'C%d' % r2, lw = 1,
                        label = 'CR%d' % (regime))
        ax[r1].grid()
        ax[r1].legend(ncol = 2, loc = 2)

    ax[0].set_ylim([-0.2, 4.2])
    ax[0].set_yticks([0, 2, 4])
    ax[0].tick_params(axis = 'x', bottom = False, labelbottom = False)
    ax[1].set_xlabel('ranked subgrid cell')
    plt.figtext(0.07, 0.5, 'avg. precip. rate (mm / h)', va = 'center',
                rotation = 90)

    for r1, regimeset in enumerate((regimes1, regimes2)):
        for r2, regime in enumerate(regimeset):
            frac = rankNumNonzero[regime] / count[regime]
            ax[r1 + 2].plot(range(1, npixel + 1), frac, color = 'C%d' % r2, 
                            lw = 1, label = 'CR%d' % (regime))
        ax[r1 + 2].grid()
        ax[r1 + 2].legend(ncol = 2, loc = 2)

    ax[2].set_ylim([-0.05, 1.05])
    ax[2].tick_params(axis = 'x', bottom = False, labelbottom = False)
    ax[3].set_xlabel('ranked subgrid cell')
    plt.figtext(0.49, 0.5, 'precip. freq.', va = 'center', rotation = 90)

    fig.savefig('fig.ranked.pdf')
    plt.close()


#--- Diurnal figures ---#


if 'diurnal.avg' in options:

    numNonzero = get_processed_data('numNonzero', year0, month0, 
                                    year1, month1, tres)
    meanPrecip = get_processed_data('meanPrecip', year0, month0, 
                                    year1, month1, tres)

    avgPrec, avgFrac = {}, {}
    for regime in regimes:
        for sat in ('A', 'T'):
            avgPrec[sat, regime] = np.mean(meanPrecip[sat, regime])
            avgFrac[sat, regime] = np.mean(numNonzero[sat, regime]) / npixel

    fig = plt.figure(figsize = (dcol, 0.75 * scol))

    ax = fig.add_subplot(121)
    ax.bar(np.arange(12) - 0.2, [avgPrec['T', regime] for regime in regimes], 
           fc = 'C0', width = 0.4, label = 'Terra')
    ax.bar(np.arange(12) + 0.2, [avgPrec['A', regime] for regime in regimes], 
           fc = 'C3', width = 0.4, label = 'Aqua')
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels(['%d' % regime for regime in regimes])
    ax.set_xlabel('CR')
    ax.set_ylim([0, 1])
    ax.set_ylabel('avg. precip. rate (mm / h)')
    ax.legend()
    ax.grid()

    ax = fig.add_subplot(122)
    ax.bar(np.arange(12) - 0.2, [avgFrac['T', regime] for regime in regimes], 
           fc = 'C0', width = 0.4, label = 'Terra')
    ax.bar(np.arange(12) + 0.2, [avgFrac['A', regime] for regime in regimes], 
           fc = 'C3', width = 0.4, label = 'Aqua')
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels(['%d' % regime for regime in regimes])
    ax.set_xlabel('CR')
    ax.set_ylabel('avg. precip. frac.')
    ax.grid()

    fig.savefig('fig.diurnal.avg.pdf')
    plt.close()


if 'diurnal.ranked' in options:

    rankSumPrecip = get_processed_data('rankSumPrecip', year0, month0, 
                                       year1, month1, tres)
    count = get_processed_data('count', year0, month0, year1, month1, tres)

    fig = plt.figure()

    for r1, regimeset in enumerate((regimes1, regimes2)):
        ax = plt.subplot(211 + r1)
        for r2, regime in enumerate(regimeset):
            ratio = ((rankSumPrecip['A', regime] / count['A', regime]) / 
                     (rankSumPrecip['T', regime] / count['T', regime]))
            ax.plot(range(1, npixel + 1), ratio, color = 'C%d' % r2, 
                    lw = 1, label = 'CR%d' % (regime))
        ax.grid()
        ax.legend(ncol = 3, loc = 2, borderpad = 0.3, 
                  labelspacing = 0.2, columnspacing = 1)
        ax.set_yscale('log')
        ax.set_ylim([0.25, 4])
        ax.set_yticks([0.25, 0.5, 1, 2, 4])
        ax.set_yticks((), minor = True)
        ax.set_yticklabels([0.25, 0.5, 1, 2, 4])
        if r1 == 0:
            ax.tick_params(axis = 'x', bottom = False, labelbottom = False)

    plt.figtext(0.5, 0.0, 'ranked subgrid cell', ha = 'center', va = 'top')
    plt.figtext(-0.01, 0.5, 'Aqua-Terra ratio', va = 'center', rotation = 90)

    fig.savefig('fig.diurnal.ranked.pdf')
    plt.close()
