This repository provides the codes used in the paper *Subgrid Precipitation Properties of Mesoscale Atmospheric Systems Represented by MODIS Cloud Regimes* by Jackson Tan and Lazaros Oreopoulos, published in the Journal of Climate (2019), doi: 10.1175/JCLI-D-18-0570.1.

The code `composite.hhr.py` composites IMERG precipitation data with MODIS CRs for each month and outputs a file containing the processed data. The code `plot.py` produces the various plots used in the paper. Both codes require functions defined in `func.py`. To run the codes, the paths to the MODIS CR, the MODIS UTC time, and the IMERG data have to be specified.

These codes have been tested with Python 3.6.5 on Ubuntu 16.04.4.

Jackson Tan
29 Jan 2019
