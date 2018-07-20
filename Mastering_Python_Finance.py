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
import math

pd.set_option('display.width', 500)
pd.options.display.float_format = '{:,.4f}'.format
pd.options.display.max_rows = 10
np.set_printoptions(linewidth=200)



def zero_coupon_bond(par, y, t):
    """
    Price a zero coupon bond.
    Par - face value of the bond.
    y - annual yield or rate of the bond.
    t - time to maturity in years.
    """
    return par/(1+y)**t


class BootstrapYieldCurve(object):
    """ Bootstrapping the yield curve """
    def __init__(self):
        self.zero_rates = dict() # Map each T to a zero rate
        self.instruments = dict() # Map each T to an instrument

    def add_instrument(self, par, T, coup, price,compounding_freq=2):
        """ Save instrument info by maturity """
        self.instruments[T] = (par, coup, price, compounding_freq)
        
    def get_zero_rates(self):
        """ Calculate a list of available zero rates """
        self.__bootstrap_zero_coupons__()
        self.__get_bond_spot_rates__()
        return [self.zero_rates[T] for T in self.get_maturities()]
    
    def get_maturities(self):
        """ Return sorted maturities from added instruments. """
        return sorted(self.instruments.keys())
        
    def __bootstrap_zero_coupons__(self):
        """ Get zero rates from zero coupon bonds """
        for T in self.instruments.iterkeys():
            (par, coup, price, freq) = self.instruments[T]
            if coup == 0:
                self.zero_rates[T] = self.zero_coupon_spot_rate(par, price, T)
    
    def __get_bond_spot_rates__(self):
        """ Get spot rates for every maturity available """
        for T in self.get_maturities():
            instrument = self.instruments[T]
            (par, coup, price, freq) = instrument
            if coup != 0:
                self.zero_rates[T] = self.__calculate_bond_spot_rate__(T, instrument)
                
    def __calculate_bond_spot_rate__(self, T, instrument):
        """ Get spot rate of a bond by bootstrapping """
        try:
            (par, coup, price, freq) = instrument
            periods = T * freq # Number of coupon payments
            value = price
            per_coupon = coup / freq # Coupon per period
            for i in range(int(periods)-1):
                t = (i+1)/float(freq)
                spot_rate = self.zero_rates[t]
                discounted_coupon = per_coupon * \
                math.exp(-spot_rate*t)
                value -= discounted_coupon
            last_period = int(periods)/float(freq)
            spot_rate = -math.log(value /
            (par+per_coupon))/last_period
            return spot_rate

        except:
            print( "Error: spot rate not found for T=%s" % t)

    def zero_coupon_spot_rate(self, par, price, T):
        """ Get zero rate of a zero coupon bond """
        spot_rate = math.log(par/price)/T
        return spot_rate

        
class ForwardRates(object):
    """
    Get a list of forward rates
    starting from the second time period
    """
    def __init__(self):
        self.forward_rates = []
        self.spot_rates = dict()
    
    def add_spot_rate(self, T, spot_rate):
        self.spot_rates[T] = spot_rate
    
    def __calculate_forward_rate___(self, T1, T2):
        R1 = self.spot_rates[T1]
        R2 = self.spot_rates[T2]
        forward_rate = (R2*T2 - R1*T1)/(T2 - T1)
        return forward_rate

    def get_forward_rates(self):
        periods = sorted(self.spot_rates.keys())
        for T2, T1 in zip(periods, periods[1:]):
            forward_rate = self.__calculate_forward_rate___(T1, T2)
            self.forward_rates.append(forward_rate)
        return self.forward_rates


def bond_ytm(price, par, T, coup, freq=2, guess=0.05):
    """ Get yield-to-maturity of a bond
        To solve for YTM is typically a complex process, and most bond
        YTM calculators use Newton's method as an iterative process."""
    """NOTE: make use this function works in real life problem"""
    freq = float(freq) # Don't think this is required for Python 3 and above anymore as all divisions will result in float numbers
    periods = T * freq
    coupon = (coup / 100) * (par / freq)
    dt = [(i+1)/freq for i in range(int(periods))]
    ytm_func = lambda y: (sum([coupon / (1 + y/freq)**(freq * t) for t in dt])
                          + par / (1 + y/freq)**(periods) - price)
    # returning the YTM function will not provide any value
    # we still need to evaluate it
    return sco.newton(func=ytm_func, x0=guess)
    
def bond_price(ytm, par, T, coup, freq=2):
    """ Get bond price from YTM """
    freq = float(freq) # Don't think this is required for Python 3 and above anymore as all divisions will result in float numbers
    periods = T * freq
    coupon = (coup / 100) * (par / freq)
    dt = [(i+1)/freq for i in range(int(periods))]
    price  = (sum([coupon / (1 + ytm/freq)**(freq * t) for t in dt])
                          + par / (1 + ytm/freq)**(periods))
    return price

def bond_mod_duration(price, par, T, coup, freq, dy=0.01):
    """ Calculate modified duration of a bond """
    """
    Some duration
    measures are: effective duration, Macaulay duration, and modified duration.
    The type of duration that we will discuss is modified duration, which measures
    the percentage change in bond price with respect to a percentage change in yield
    (typically 1 percent or 100 basis points (bps) DV01).
    The higher the duration of a bond, the more sensitive it is to yield changes.
    Conversely, the lower the duration of a bond, the less sensitive it is to yield changes.
    duration describes the linear price-yield relationship for a small change in Y.
    Because the yield curve is not linear, using a large value of dy does
    not approximate the duration measure well.
    """
    
    ytm = bond_ytm(price, par, T, coup, freq, guess=0.01)
    price_neg = bond_price(ytm=(ytm - dy), par=par, T=T, coup=coup, freq=freq)
    price_pos = bond_price(ytm=(ytm + dy), par=par, T=T, coup=coup, freq=freq)
    
    return (price_neg - price_pos) / (2 * price * dy)
    

def bond_convexity(price, par, T, coup, freq, dy=0.01):
    """
    Convexity is the sensitivity measure of the duration of a bond to yield changes.
    Convexity as a risk management tool to measure the amount of
    market risk in the portfolio. Higher convexity portfolios are less affected by interest
    rate volatilities than lower convexity portfolio, given the same bond duration and
    yield. As such, higher convexity bonds are more expensive than lower convexity
    ones, everything else being equal.
    Higher convexity bonds will exhibit higher price changes for the same change
    in yield.
    """
    ytm = bond_ytm(price, par, T, coup, freq, guess=0.01)
    price_neg = bond_price(ytm=(ytm - dy), par=par, T=T, coup=coup, freq=freq)
    price_pos = bond_price(ytm=(ytm + dy), par=par, T=T, coup=coup, freq=freq)
    
    return (price_neg + price_pos - 2*price) / (price * dy**2)

def one_factor_vasicek(r0, K, theta, sigma, T=1., N=10, seed=777):
    """ Simulate interest rate path by the Vasicek model """
    """
    In the one-factor Vasicek model, the short rate is modeled as a
    single stochastic factor
    From the book:
        The Vasicek follows an Ornstein-Uhlenbeck process,
        where the model reverts around the mean  with K, the speed of
        mean reversion. As a result, the interest rates may become negative,
        which is an undesirable property in most normal economic conditions.
        
        Really? Nowadays is reality.
    r0 : initial rate of interest at t=0
    T : period in terms of number of years
    N is the number of intervals for the modeling process
    sigma : instantaneous standard deviation
    W(t) : random Wiener process
    seed is the initialization value for NumPy's standard normal random number generator.
    
    Question: How to design a vectorized Vasicek Model?
    """
    np.random.seed(seed)
    dt = T / N
    rates = [r0] # makes it a list
    for i in range(N):
        dr = (K * (theta - rates[-1] * dt)
              + sigma * np.random.normal())
        rates.append(rates[-1] + dr)
    return range(N+1), rates


def cox_ingersoll_ross(r0, K, theta, sigma, T=1.,N=10,seed=777):
    """ Simulate interest rate path by the CIR model """
    """
    The Cox-Ingersoll-Ross (CIR) model is a one-factor model that was proposed
    to address the negative interest rates found in the Vasicek model.
    """
    np.random.seed(seed)
    dt = T / N
    rates = [r0] # makes it a list
    for i in range(N):
        dr = (K * (theta - rates[-1] * dt)
              + sigma * np.sqrt(rates[-1]) * np.random.normal())
        rates.append(rates[-1] + dr)
    return range(N+1), rates


def rendleman_bartter(r0, theta, sigma, T=1.,N=10,seed=777):
    """ Simulate interest rate path by the Rendleman-Barter model """
    """
    Here, the instantaneous drift is theta*r(t) with an instantaneous
    standard deviation sigma*r(t). The Rendleman and Bartter model can
    be thought of as a geometric Brownian motion, akin to a stock price
    stochastic process that is log-normally distributed. This model lacks
    the property of mean reversion. Mean reversion is a phenomenon where
    the interest rates seem to be pulled back toward a long-term average level.
    In general, this model lacks the property of mean reversion and grows
    toward a long-term average level.
    """
    np.random.seed(seed)
    dt = T / N
    rates = [r0] # makes it a list
    for i in range(N):
        dr = ((theta * rates[-1] * dt)
              + sigma * rates[-1] * np.random.normal())
        rates.append(rates[-1] + dr)
    return range(N+1), rates


def brennan_schwartz(r0, K, theta, sigma, T=1., N=10, seed=777):
    """ Simulate interest rate path by the Brennan Schwartz model """
    """
    The Brennan and Schwartz model is a two-factor model where the short-rate
    reverts toward a long rate as the mean, which also follows a stochastic process.
    """
    np.random.seed(seed)
    dt = T / N
    rates = [r0] # makes it a list
    for i in range(N):
        dr = (K*(theta-rates[-1])*dt
              + sigma*rates[-1]*np.random.normal())
        rates.append(rates[-1] + dr)
    return range(N+1), rates

def exact_zcb(theta, kappa, sigma, tau, r0=0.):
    """ Get zero coupon bond price by Vasicek model """
    # tau = T - t
    B = (1 - np.exp(-kappa*tau)) / kappa
    A = np.exp((theta-(sigma**2)/(2*(kappa**2))) * (B-tau) - (sigma**2)/(4*kappa)*(B**2))
    return A * np.exp(-r0 * B)


def exercise_value(K, R, t):
    """
    Implementation of the early-exercise option
    k is the price ratio of the strike price to the par value
    r is the interest rate for the strike price.
    """
    return K * np.exp(-R * t)


































































































