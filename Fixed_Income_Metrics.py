
# coding: utf-8

# In[4]:

import numpy as np
import scipy.interpolate as spi
import matplotlib.pyplot as plt



# In[60]:

#Par Rates
#yield_curve2 = {1:0.06, 2:0.08, 3:0.095, 4:0.105, 5:0.11, 6:0.1125, 7:0.1138, 8:0.1144, 9:0.1148, 10:0.115}
#yield_curve2 = {1:0.04, 2:0.045, 3:0.05, 4:0.055}
#yield_curve2 = {0.5:0.03, 1:0.033, 2:0.039, 3:0.047, 4:0.05, 5:0.052, 6:0.054, 7:0.0555, 8:0.0565, 9:0.058, 10:0.06}
#yield_curve2 = {1:0.05, 2:0.055, 3:0.05, 4:0.06}
'''yield_curve2 = {0.25:0.0101,0.50:0.0251,0.75:0.0301,1:0.0398,2:0.04006,3:0.04001,4:0.03995,5:0.03988,
                6:0.03988,7:0.03991,8:0.03999,9:0.04011,10:0.04025,
                12:0.04055,15:0.04094,20:0.04124,25:0.04122,30:0.04102
               }'''

'''yield_curve2 = {1:0.0398,2:0.04006,3:0.04001,4:0.03995,5:0.03988,
                6:0.03988,7:0.03991,8:0.03999,9:0.04011,10:0.04025,
                12:0.04055,15:0.04094,20:0.04124,25:0.04122,30:0.04102
               }'''
#yield curve taken from spreadsheet of TM Week 9
'''yield_curve2 = {1./12:0.02638, 2./12:0.02679, 3./12:0.02741, 4./12:0.02799, 5./12:0.0285, 6./12:0.02904,
                7./12:0.02954, 8./12:0.02994, 9./12:0.03033, 10./12:0.03075, 11./12:0.03106, 1:0.03135, 2:0.0339,
                3:0.03493, 4:0.03564, 5:0.0362, 6:0.0367, 7:0.03718, 8:0.03764, 9:0.03808, 10:0.03847}'''

'''yield_curve_live = {1:-0.0035, 2:-0.00221, 3:-0.00216, 4:-0.00194, 5:-0.00148, 6:-0.0008, 7:0.00004, 8:0.00098,
                   9:0.00191, 10:0.00278, 12:0.00428, 15:0.00588, 20:0.00713, 25:0.00741, 30:0.00742}'''

#yield_gbp = {0.5:0.0029, 1:0.0016, 2:0.0014, 5:0.0022}
#yield_usd = {0.5:0.0045, 1:0.0059, 2:0.0075, 5:0.0115}
yield_eur = {0.5:-0.006331, 1:-0.006378, 2:-0.006516, 5:-0.005373}

bond_parameters = {'Coupon':0.05,
                   'Frequency':1.0,
                   'Maturity':5.0}


# In[41]:

#Generating discount factors from the par yield curve and Bootstrapping
#this is done in two steps: 1st--> separate the curve into short-term (<1yr) and long-term (>=1yr)
#2nd--> for short-term curve calculate the DF using simple interest and for long-term curve use compounding interest
def discount_factor_spot_forward(yield_curve):
    df = {}
    #short-term yield curve <1yr
    #The separation of short and long-term yield curve is necessary for interpolation of the longer dates
    short_rates = { k:v for k, v in yield_curve.items() if k < 1 }#creating a dic where all Keys are <1
    for key in short_rates:        
        #Note: as the rates are very short-term we can use this simple formula to calculate the DF for periods up to 1yr
        d = 1.0 / (1.0+(key * short_rates[key]))
        df[key]= d
    
    #perform an interpolation of the yield curve because there will be gaps in longer maturities
    #this can only be done by maturities starting at 1yr
    long_rates = { k:v for k, v in yield_curve.items() if k >= 1 }#creating a dic where all Keys are >=1
    tm = np.array(long_rates.keys())#time to maturity
    yr = np.array(long_rates.values())#rates per maturity
    interp = spi.interp1d(tm,yr,)#linear interpolation is done here
    
    curve_par = {}
    for key in range(1,tm.max()+1):#creating the par curve from the interpolation
        value = float(interp(key))
        curve_par[key] = value
    
    maturities = []#empity list created to allocate the yield curve maturities. This is done to avoid entering the longer maturities, making the process more automated
    for mat in curve_par:
        maturities.append(mat)
    
    #once interpolation is done, we'll create now the discount factors from 1yr to 30yr using an iteration in "lista"
    df_lst = {}
    for maturity in maturities:#this is the Bootstrapping process
        if maturity==1:
            df1 = 1./(1.+curve_par[maturity])
            df_lst[maturity] = df1
        else:
            somation=0
            for j in range(1,maturity):
                somation = somation + (curve_par[maturity] * df_lst[j])
            df_n = (1. - somation)/(1. + curve_par[maturity])
            df_lst[maturity] = df_n
    df.update(df_lst)
    
    #SPOT RATES
    #once the discount factors have been calculated we can use them to find the Spot rates, which should give us the same
    #price as per DF
    #Calculating the Spot Yield Curve
    s_rates = {}#empty dic created to receive all the spot rates within the for-loop below
    for factor in df:
        spot_n = (1.0/df[factor])**(1.0/factor) - 1
        s_rates[factor] = spot_n
    
    #FORWARD RATES
    #Implied Spot Rate 1yr Forward: F1,2; F1,3; F1,4
    #1yr Forward Rate: F1,2; F2,3; F3,4 --> this is the rate the traders use
    #Note: the Spot Rate = F0,n
    #Note: it must also be broken into two curves: short-term (simple interest) and long-term (compounding interest)
    fwd_rates = {}
    for spot in s_rates:
        n = spot
        m = n-1
        if n>=1:
            if m==0:
                fwd_rates[spot] = s_rates[n]
            elif n in s_rates.keys():
                #1yr Forward Rate: F1,2; F2,3; F3,4 --> this is the rate the traders use
                fwd = (((1+s_rates[n])**n) / ((1+s_rates[m])**m))**(1/(n-m)) - 1
                fwd_rates[spot] = fwd
    return curve_par,df, s_rates,fwd_rates


# In[66]:

Par, DF, Spot, Fwd = discount_factor_spot_forward(yield_eur)
Spot


# In[63]:

plt.plot(Fwd.keys(), Fwd.values())
plt.plot(Spot.keys(), Spot.values())
plt.plot(Par.keys(), Par.values())
plt.show()


# In[8]:

#PV01 calculated using the Spot Curve
#it gives the bond's sensitivity to interest rates
def pv01(spot_curve,coupon,t,par):
    up_curve = {}#new spot curve after addition of "X"basis points to each rate on the curve
    down_curve = {}#new spot curve after subtraction of "X"basis points to each rate on the curve
    bps = 0.0001
    #bumpping the spot curve Up and Down by "bps"
    for maturity in spot_curve:
        up_bump = spot_curve[maturity] + bps
        down_bump = spot_curve[maturity] - bps
        up_curve[maturity] = up_bump
        down_curve[maturity] = down_bump
    
    #Price Up
    #calculation done only for bonds with maturity >=1yr
    pv_cp_up = 0
    for i in range(1,t+1):
        pv_cp_up+= coupon*par/((1+up_curve[i])**i)#present values of the coupons
        pv_princ_up = par/((1+up_curve[t])**t)#present value of face value at maturity
    price_up = pv_cp_up + pv_princ_up#present value of the bond
        
    #Price Down
    #calculation done only for bonds with maturity >=1yr
    pv_cp_down = 0
    for i in range(1,t+1):
        pv_cp_down+= coupon*par/((1+down_curve[i])**i)#present values of the coupons
        pv_princ_down = par/((1+down_curve[t])**t)#present value of face value at maturity
    price_down = pv_cp_down + pv_princ_down#present value of the bond
    pv01 = (price_down - price_up)/2#calculation of PV01
    return price_up, price_down, pv01


# In[ ]:




# In[9]:

#Testing Bond pricing. A the par bond priced using its on coupon rate should produce a value=100
#The same Bond should produce a value=100 if we use the discount factors

def bond_price_par_rate(coupon,maturity,par):
    #this part can only be used to test the price of a par bond that's in the curve
    #Bond Priced using the par bond and Par Rate
    soma = 0
    rate = coupon
    for i in range(1,maturity+1):
        if i<mat:
            soma += coupon*par/(1 + rate)**i#note that the rate never changes with changing maturities
        else:
            soma += (par + coupon*par)/(1 + rate)**i
    print 'Bond Priced using the par bond and Par Rate: ',soma

def bond_price_disc_factors(coupon,maturity,par,disc_factors):
    #Bond Priced using the par bond and Discount Factors
    soma2 = 0
    for i in range(1,maturity+1):
        if i<maturity:
            soma2 += (coupon*par)*disc_factors[i]
        else:
            soma2 += (par + coupon*par)*disc_factors[i]
    print 'Bond Priced using the par bond and Discount Factors: ',soma2


def bond_price_spot_curve(coupon,maturity,par,spot_curve):
    #Testing to find the bond price from the Spot Rate Curve
    #It must produce the same results as per Discount Factors
    pv_cp = 0
    for i in range(1,maturity+1):
        pv_cp+= coupon*par/((1+spot_curve[i])**i)
        #print pv_cp
        pv_princ = par/((1+spot_curve[maturity])**maturity)
    price = pv_cp + pv_princ
    print 'Bond Priced using the par bond and Spot Rates: ',price


# In[11]:

'''df, spot, fwd = discount_factor_spot_forward(yield_curve_live)
cp = 0.00428
mat = 12
par = 100
bond_price_par_rate(cp,mat,par),
bond_price_disc_factors(cp, mat,par,df ),
bond_price_spot_curve(cp,mat,par,spot)
pv01(spot, cp,mat,par)'''


# In[12]:

'''d,s,f = discount_factor_spot_forward(yield_curve_live)
plt.plot(s.values())
plt.show()'''


# In[ ]:




# In[13]:

"""The Swap Curve"""


# In[ ]:



