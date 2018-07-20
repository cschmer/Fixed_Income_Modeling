from __future__ import division
import numpy as np
import scipy as sp
import scipy.stats as st
import pandas as pd
import pandas_datareader.data as web
from datetime import datetime, date, time
from dateutil.parser import parse
import csv
import json
import openpyxl
from openpyxl import load_workbook
import matplotlib.pyplot as plt
from urllib import urlretrieve
from random import gauss
import pickle
from math import *
import numba as nb

pd.set_option('display.width', 500)
pd.options.display.float_format = '{:.2f}'.format
pd.options.display.float_format = '{:,}'.format
pd.options.display.max_rows = 10
np.set_printoptions(linewidth=200)

#=======================================================================================================================================================================================================
""" Notes
First: Develop the "Term Structure Modeling with No-Arbitrage Interest Rate Models
There will be also model called "Equilibrium Models"
One factor model represent the short of interest, whereas multifactor models incorporate a SDE for additional interest rates
Single Factor model that assumes a stationary variance or, as it is more often called, volatility.
The lattice is a representation of the model, capturing the distribution of rates over time.

First: the lattice is used to generate cash flows over the life of the security
Second: the interest rates on the lattice are used to compute the present value of those cash flows.

The interest tree generated must produce a value for an on-the-run optionless issue that is consistent
with the current par yield curve. The generated value must equal to the observed market price for the
optionless instrument. A lattice that produces it is said to be "fair". The lattice can only be used for
valuation when it has been calibrated to be fair. The lattice must be calibrated to the current par yield curve.
Ultimately, the lattice must price optionless par bonds at par.

Assumptions: rates at any point in time have the same probability of occurring; in other words, the probability is 50% on each leg.

Interest tree model is based on a log-normal random walk with known stationary Volatility
For an interest model to be useful for practical purposes, it is helpful to adapt them to a recombining lattice structure. This usually
translates into a binomial or trinomial trees. 
"""
#=======================================================================================================================================================================================================


#=======================================================================================================================================================================================================
"""        *** Ho-Lee Model***        """


#=======================================================================================================================================================================================================





























































































































































