
# coding: utf-8

# In[1]:

import numpy as np
import scipy.interpolate as spi
import matplotlib.pyplot as plt


# In[50]:

#Bond data
p = 1000000 #principal
fv = p+(p*0.045*0.5)
rates_simple = {0.25:[0.06, 0.0401, 0.0399],
         0.50:[0.06, 0.0451, 0.0449],
         0.75:[0.06, 0.0501, 0.0499],
         1.00:[0.06, 0.0601, 0.0599]}


# In[65]:

#Generating Discount Factors(under simple intesrest) for less than 1yr for PV01 calculation
#PV01 for short rate instruments (<1yr)
#The PV01 is the forward value of the liability, multiplied by the
#difference in the two adjusted levels, divided by 2
#pv01 of loan/liability is negative, of an asset is positive
#NOTE: To determine the PV01 of a portfolio of cashflowsin the future,
#      sum up the PV01’s of the individual cashflows
def simple_interest_disc_factor(par_rates,fut_value):
    df_simple = {}
    pv01_simple = 0
    for period in par_rates:
        lst=[]
        for par_rate in par_rates[period]:
            df = 1.0 / (1.0+(par_rate*period))
            lst.append(df)
        df_simple[period]= lst
    pv01_simple = -fut_value * (df_simple[0.5][1] - df_simple[0.5][2])/2
    pv01mm_simple = (df_simple[0.5][1] - df_simple[0.5][2])/2.0 * -1000000
    return fut_value, pv01mm_simple, pv01_simple, df_simple
    


#This is for a par yield rate as it won't change with time
#Note that the discount factors can only be used to price bonds with the same maturity
def par_compound_interest_disc_factor(rates_compound,maturity):
    df_compound = {}
    for period in range(1,maturity+1):
        df_compound_n = 1.0/((1.0+rates_compound)**period)
        df_compound[period] = df_compound_n
    return df_compound





# In[67]:

simple_interest_disc_factor(rates_simple,fv)


# In[ ]:



