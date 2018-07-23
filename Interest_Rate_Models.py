from __future__ import division
import numpy as np
import scipy as sp
import scipy.stats as st
from scipy.optimize import fsolve
import pandas as pd
from datetime import datetime, date, time
import matplotlib.pyplot as plt
import math
import numba as nb
import copy

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
"""        *** Black-Derman-Toy "BDT" Model***        """
"""
    The model is obtained from the equation stochastic equation below:
    
            df(r(t)) = theta(t) + rho(t)*g(r(t))dt + sigma(r(t), t)dz    -------> Beutow-Hanke-Fabozzi
    
    f and g : chosen functions of the short-term rate and are the same for most no-arbitrage models
    theta : drift of the short-term rate
    rho : mean reversion term to an equilibrium rate.
    sigma : local volatility of the short-term rate
    z : normally distributed Wiener process that captures the randomness of future changes in the short-term rate.
    
    With f(r) = ln(r) and g(r) = ln(r), the BDT model is generated:
    
            dln(r) = theta(t) + rho(t)ln(r)dt + sigma(r(t),t)dz
            
                    rho(t) = sigma'(t) / sigma(t)
            
        ***    dln(r) = [theta(t) + (sigma'(t)/sigma(t))*ln(r)]dt + sigma(t)dz    ***
The 3 key features are:

    1. Its fundamental variable is the short-term rate - the annualized one-period interest rate. The short rate is
    the one-factor of the model; its changes drive all security prices;
    2. The model takes as inputs an array of long rates (yields on zero-coupon Treasury bonds) for various maturities
    and an array of yield volatilities for the same bonds. The first array is called the yield curve and the second the
    volatility curve. Together these curves form the Term Structure.
    3. The model varies an array of mean and an array of volatilities for the future short rate to match the inputs. As the
    future volatility chabges, the future mean reversion changes. 

"""

#=======================================================================================================================================================================================================


#===============================================================================
# Helper function just to print the lattice
def print_lattice(lattice, info=[]):
    """
    Lattice Printer
    """
    levels = len(lattice[-1])
    start_date = len(lattice[0]) - 1
    dates = levels - start_date
    outlist = []  # no fucking idea why that yet
    col_widths = [0] * dates  # list of zeros
    for j in range(dates):
        level = []
        for k in range(dates):
            try:
                point = "{:.16f}".format(lattice[k][levels - 1 - j])
                esc_width = 0
                if info != [] and info[k][levels - 1 - j] > 0:
                    point = (point, 'red')
                    esc_width += 9
                level.append(point)
                col_widths[k] = max(col_widths[k], len(point) - esc_width)
            except IndexError:
                level.append('')
        outlist.append(level)
    separator = "|-".join(['-' * w for w in col_widths])
    formats = []
    for k in range(dates):
        formats.append("%%%ds" % col_widths[k])
    pattern = " ".join(formats)
    print(pattern % tuple(str(start_date + time) for time in range(dates)))
    print(separator)
    for line in outlist:
        print(pattern % tuple(line))
#===============================================================================

#===============================================================================
# Implement a Function to help generate the Multipliers or Factors


def factor(time):
    """
    Helper function to generate the multipliers
    """
    factor = [[]]  # List of lists with numbers increasing by a factor equal 2
    temp = []
    for x in range(time):
        if x == 0:
            factor[0].append(0)
        else:
            for y in range(0, x + 1):
                if y == 0:
                    temp.append(1)
                else:
                    temp.append(y * 2)
                    x = x - 2
            factor.append(temp)
            temp = []
    return factor
                    
#===============================================================================


#===============================================================================
# Testing helper functions
multiplier = factor(5)

multiplier = [list(reversed(x)) for x in multiplier][:]
m_guess = 0.2  # initial guess for the second node
nNodes = 5
observedRates = [0.10, 0.11, 0.12, 0.125, 0.13]
vol = [0.20, 0.19, 0.18, 0.1622, 0.14365]

zCBond_0 = 100 / (1 + observedRates[0]) ** 1
zCBond_1 = 100 / (1 + observedRates[1]) ** 2
zCBond_2 = 100 / (1 + observedRates[2]) ** 3
zCBond_3 = 100 / (1 + observedRates[3]) ** 4
zCBond_4 = 100 / (1 + observedRates[4]) ** 5

pv = [zCBond_0, zCBond_1, zCBond_2, zCBond_3, zCBond_4]

temp_rate_tracker = [[observedRates[0]]]


def bdtONE(guess):
    ru = guess * math.exp(2 * vol[1])
    rd = guess
    N1 = (100) / (1 + ru)
    N2 = (100) / (1 + rd)
    return (0.5 * ((N1 / (1 + observedRates[0])) + (N2 / (1 + observedRates[0]))) - zCBond_1)


g = fsolve(bdtONE, m_guess)[0]

ru = g * math.exp(2 * vol[1])
rd = g

m = [0.1, 0.1]  # Guess list data structure for node 2 and above

final_rate_tracker = [[]]
temp_rate_tracker = [[observedRates[0]], [ru, rd]]

mo = [[0.1], [rd, 1]]  # 1 means first node
m = [0.1, 0.1]

final_rate_tracker = temp_rate_tracker[1:]


def rateCalculator(m, node):
    rate = []
    for x in range(0, node + 1):
        if x == node:
            rate.append(m[0])
        else:
            rate.append(m[0] * math.exp(multiplier[node][x] * m[1]))
    return rate


def valueCalculator(m, node, final_rate_tracker):
    nValue = []
    nValue2 = []
    rate = []
    final_rate_tracker = final_rate_tracker[:]
    final_rate_tracker.reverse()
    for x in range(node + 1):
        if x == node:
            rate.append(float(m[0]))
        else:
            rate.append(m[0] * math.exp(multiplier[node][x] * m[1]))
    
    for x in range(len(rate)):
        nValue2.append(((100) / (1 + rate[x])))
        nValue = copy.deepcopy(nValue2)
    
    nValue2 = []
    y = 0
    
    while len(nValue) > 2:
        if len(nValue) <= 2:
            break
        
        for x in range(len(nValue) - 1):
            nValue2.append((0.5 * ((nValue[x]) + (nValue[x + 1]))) / (1 + final_rate_tracker[y][x]))
        
        if y == 0:
            yu = math.sqrt(100 / nValue2[0]) - 1
            yd = math.sqrt(100 / nValue2[1]) - 1
        rate = []
        y = y + 1
        nValue = nValue2[:]
        nValue2 = []
        
    out = [(0.5 * ((nValue[0]) + (nValue[1])) / (1 + temp_rate_tracker[0][0]) - pv[node])]
    out.append((0.5 * math.log(yu / yd) - vol[node]))
    return out


def solutionIterator(mo, nNodes, final_rate_tracker, temp_rate_tracker):
    
    for x in range(2, nNodes):
        data = (x, final_rate_tracker)
        mo.append(fsolve(valueCalculator, m, args=data))
        temp_rate_tracker.append(rateCalculator(mo[x], x))
        final_rate_tracker = temp_rate_tracker[1:]
        data = (x, final_rate_tracker)
    return temp_rate_tracker


final_rate_tracker = solutionIterator(mo, nNodes, final_rate_tracker, temp_rate_tracker)[:]

reversed_lists = [list(reversed(x)) for x in final_rate_tracker]

finalLattice = [[100 * y for y in x] for x in reversed_lists]
print_lattice(finalLattice, info=[])

