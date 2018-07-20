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



""" Notes
First: Develop the "Term Structure Modeling with No-Arbitrage Interest Rate Models
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



"""
# 
# def Binomial_Interest_Tree(t,steps,sigma,r):
#     """
#     t     : time to maturity
#     dt    : length of time steps
#     p     : probability "martingale probability"
#     r     : risk free rate
#     sigma : volatility
#     a     : growth factor per step or discount factor per time interval
#     u     : up step size
#     d     : down step size
#     """
#     dt = t/steps
#     print(dt)
#     u = sigma * np.sqrt(dt)
#     print(u)
#     #d = 1/u
#     #a = np.exp(r * dt)
#     p = 0.5 + (r*np.sqrt(dt) / (2*sigma))
#     print(p)
#     S0 = 100
#     M = steps
#     mu = np.arange(M + 1)
#     mu = np.resize(mu, (M + 1, M + 1))
#     md = np.transpose(mu)
#     mu = p ** (mu - md)
#     md = (1-p) ** md
#     S = S0 * mu * md
#     
# 
# Binomial_Interest_Tree(1,5,0.20,0.05)
# 
# 
# 
# 
# 
# 
# def binomial_py(strike):
#     ''' Binomial option pricing via looping.
#     
#     Parameters
#     ==========
#     strike : float
#         strike price of the European call option
#     '''
#     # LOOP 1 - Index Levels
#     S = np.zeros((M + 1, M + 1), dtype=np.float64)
#       # index level array
#     S[0, 0] = S0
#     z1 = 0
#     for j in range(1, M + 1, 1):
#         z1 = z1 + 1
#         for i in range(z1 + 1):
#             S[i, j] = S[0, 0] * (u ** j) * (d ** (i * 2))
#             
#     print(S)
#     # LOOP 2 - Inner Values
#     iv = np.zeros((M + 1, M + 1), dtype=np.float64)
#       # inner value array
#     z2 = 0
#     for j in range(0, M + 1, 1):
#         for i in range(z2 + 1):
#             iv[i, j] = max(S[i, j] - strike, 0)
#         z2 = z2 + 1
#         
#     # LOOP 3 - Valuation
#     pv = np.zeros((M + 1, M + 1), dtype=np.float64)
#       # present value array
#     pv[:, M] = iv[:, M]  # initialize last time point
#     z3 = M + 1
#     for j in range(M - 1, -1, -1):
#         z3 = z3 - 1
#         for i in range(z3):
#             pv[i, j] = (q * pv[i, j + 1] +
#                         (1 - q) * pv[i + 1, j + 1]) * df
#     return pv[0, 0]
# 
# 
# 
# 
# def binomial_np(strike):
#     ''' Binomial option pricing with NumPy.
#     
#     Parameters
#     ==========
#     strike : float
#         strike price of the European call option
#     '''
#     # Index Levels with NumPy
#     mu = np.arange(M + 1)
#     mu = np.resize(mu, (M + 1, M + 1))
#     md = np.transpose(mu)
#     mu = u ** (mu - md)
#     md = d ** md
#     S = S0 * mu * md
#     
#     # Valuation Loop
#     pv = np.maximum(S - strike, 0)
# 
#     z = 0
#     for t in range(M - 1, -1, -1):  # backwards iteration
#         pv[0:M - z, t] = (q * pv[0:M - z, t + 1]
#                         + (1 - q) * pv[1:M - z + 1, t + 1]) * df
#         z += 1
#     return pv[0, 0]
#         
#     
# S0 = 100
# T = 1
# r = 0.05
# vola = 0.20
# M = 5
# dt = 0.2
# #print(dt)
# df = np.exp(-r * dt)
# 
# u = np.exp(vola * np.sqrt(dt))
# d = 1/u
# 
# #print(q)
# #print binomial_np(100)
# # 
# # binomial_nb = nb.jit(binomial_py)
# # print round(binomial_nb(100),3)
# 
# 
# S = np.zeros((M + 1, M + 1), dtype=np.float64)
# # index level array
# S[0, 0] = S0
# z1 = 0
# for j in range(1, M + 1, 1):
#     z1 = z1 + 1
#     for i in range(z1 + 1):
#         S[i, j] = S[0, 0] * (u ** j) * (d ** (i * 2))
# 
# print(S)
# 
# q = 0.5 + (r*np.sqrt(dt) / (2*vola))
# Q = np.zeros((M + 1, M + 1), dtype=np.float64)
# Q[0, 0] = 1
# z1 = 0
# for j in range(1, M + 1, 1):
#     z1 = z1 + 1
#     for i in range(z1 + 1):
#         Q[i, j] = Q[0, 0] * (u ** j) * (d ** (i * 2))
# 
# print(S)





























 
 
class OptionClass(object):
    def __init__(self,s0,k,r,sig,t,steps=1,q=0,params={}):
        #parameters to be filled by the user
        self.s0 = s0                            #current price of underlying
        self.k = k                              #strike price
        self.r = r                              #domestic risk-free interest rate
        self.sigma = sig                        #volatility of the underlying
        self.t = t                              #time to maturity given in months
        self.steps = steps
        self.dt = float(self.t)/(12*self.steps) #time step
        self.q = q                              #div yield or foreign risk free rate
        self.is_call = params.get('call',True)  #either Put or Call -- European or American
         
        #parameters that must be calculated based on user's inputs
        self.u = np.exp(self.sigma * np.sqrt(self.dt))
        self.d = 1/self.u
        self.option_prob()
        self.terminal_stock_prices()
        self.terminal_op_payoffs()
        self.df = np.exp(-self.r*self.dt)
         
    def option_prob(self):
        if self.q<1:
            a = np.exp((self.r - self.q) * self.dt)
            prob = (a - self.d)/(self.u - self.d)
        else:
            a = 1
            prob = (a - self.d)/(self.u - self.d)
        return prob
     
    def terminal_stock_prices(self):
        tree = np.zeros(self.steps + 1)
        for i in range(self.steps + 1):
            tree[i] = self.s0*(self.u**(self.steps-i))*(self.d**(i))
        return tree
 
    def terminal_op_payoffs(self):
        payoff = np.maximum(0,(self.terminal_stock_prices() - self.k)if self.is_call 
                                else (self.k - self.terminal_stock_prices()))
        return payoff
     
    def option_price(self):
        payoff = self.terminal_op_payoffs()
        for i in range(0,self.steps):
            payoff = self.df * (payoff[:-1]*self.option_prob() + payoff[1:]*(1-self.option_prob()))
            #print payoff
        return payoff
 
 
op = OptionClass(s0=0.61,
                 k=0.60,
                 r=0.05,
                 sig=0.12,
                 t=3,
                 steps=3,
                 q=0.07,
                 params={'call':False})
#op = OptionClass(810,800,0.05,0.2,6,2,0.02,{'call':True})
#op = OptionClass(50,52,.05,.30,24,2,{'call':False})
#op = OptionClass(40,40,.04,0.30,6,2)
 
# op = OptionClass(31,30,0.05,0.30,1,2,1,{'call':False})#Put on a Future

print(op.u, op.d, op.dt, op.option_prob(),op.df)
print(op.option_price())
 
 
class OptionAmerican(OptionClass):
    def initialize_stock_price_tree(self):
        tree = np.zeros(self.steps + 1)
        for i in range(self.steps + 1):
            prev = tree[-1]
            st = np.concatenate((prev*self.u,[prev[-1]*self.d]))
            #tree[i] = self.s0*(self.u**(self.steps-i))*(self.d**(i))
            tree.append(st)
        return tree
 
 
 
def initialize_stock_price_tree(steps,u,d):
    tree = np.zeros(steps + 1)
    for i in range(steps + 1):
        prev = tree[-1]
        #print prev
        #st = np.concatenate((prev*u,prev[-1]*d))
        #tree[i]=st
    return tree
 
 
# initialize_stock_price_tree(3,1.2,0.8)

