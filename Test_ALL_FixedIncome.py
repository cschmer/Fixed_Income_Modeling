from __future__ import division, print_function
import pandas.tseries
import numpy as np
import numpy.random as npr
import scipy as sp
import scipy.stats as scs
import scipy.optimize as sco
import pandas as pd
import datetime as dt
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
import Mastering_Python_Finance as mpf
import My_Binomial_Tree as mbt

pd.set_option('display.width', 500)
pd.options.display.float_format = '{:,.4f}'.format
pd.options.display.max_rows = 10
np.set_printoptions(linewidth=200)


"""***Week 01 ***"""

par_rates = {
                1:0.035,
                2:0.042,
                3:0.047,
                4:0.052
            }

# some arbitrary values (modify at will)
sigma = 0.2
r     = 0.05
T     = 3.0
































































