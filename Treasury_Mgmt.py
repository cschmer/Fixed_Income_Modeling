# coding: utf-8

import numpy as np
import scipy.interpolate as spi
import matplotlib.pyplot as plt


"""***Week 01 ***"""

'''#generating Discount Factors(under simple intesrest) for less than 1yr for PV01 calculation
p = 1000000 #principal
fv = p+(p*0.045*0.5)
rates = {0.25:0.0401,
         0.50:0.0451,
         0.75:0.0501,
         1.00:0.0601}
def simple_interest_disc_factor(rates):
    df = {}
    for i in rates:
        lst=[]
        for j in rates[i]:
            d = 1.0 / (1.0+(j*i))
            lst.append(d)
        df[i]= lst
    return df

def compound_interest_disc_factor(r,t): #this is for a par yield rate as it won't change with time
    df = {}
    for i in range(1,t+1):
        df_n = 1.0/((1.0+r)**i)
        df[i] = df_n
    return df'''



# # In[31]:
# 
# '''#df = simple_interest_disc_factor(rates)
# compound_interest_disc_factor(0.06,4)'''
# 
# 
# # In[1223]:
# 
# '''#PV01 for short rate instruments (<1yr)
# #The PV01 is the forward value of the liability, multiplied by the
# #difference in the two adjusted levels, divided by 2
# #pv01 of loan/liability is negative, of an asset is positive
# #NOTE: To determine the PV01 of a portfolio of cashflowsin the future,
# #      sum up the PV01’s of the individual cashflows
# pv01 = -fv * (df[0.5][1] - df[0.5][2])/2
# fv, pv01'''
# 
# 
# # In[1320]:
# 
# #The PV01/mm
# #To determine the PV01 of a portfolio of cashflows in the future, sum up the PV01’s of the individual cashflows
# pv01mm_6m = (df[0.5][1] - df[0.5][2])/2.0 * -1000000
# pv01mm_6m
# 
# 
# # In[1225]:
# 
# #The PV01 increases with maturity, here given by PV01/mm
# pv01mm_3m = (df[0.25][1] - df[0.25][2])/2. * -1000000
# pv01mm_6m = (df[0.5][1] - df[0.5][2])/2. * -1000000
# pv01mm_1y = (df[1][1] - df[1][2])/2. * -1000000
# pv01mm_3m, pv01mm_6m, pv01mm_1y
# 
# 
# # In[1164]:
# 
# #P&L approx using PV01/mm
# rates_change = 50 #in bps
# profit_loss = rates_change*(-pv01mm_6m)
# profit_loss
# 
# 
# # In[1165]:
# 
# #Actual P&L calculation when rates move
# def profit_loss_actual(p,r0,r1,n):
#     #p principal of loan
#     fv = 1.+(r0*n) #FV of loan at current rates
#     fv_chg = 1.+(r1*n) #FV of loan when rates change by xbps
#     PV_int = p/fv #present value of 1mm
#     PV_chg = p/fv_chg
#     profit_loss_actual = PV_chg - PV_int
#     return profit_loss_actual
# 
# profit_loss_actual(1000000,0.045,0.05,0.5)
# 
# 
# # In[1226]:
# 
# #PV01 decreases with increasing interest rates
# p = 1000000 #principal
# fv_USD = p+(p*0.0501*1)
# fv_BRL = p+(p*0.1325*1)
# p/fv_USD,p/fv_BRL
# 
# 
# # In[1319]:
# 
# '''#FRA
# #The method of determining a forward rate  #determine simple interest forward rates
# #is the same as used for discount factors.
# #Note: For no arbitrage to be true the yield of the FRA must be calculated using the interest rate
# #      that applies between now and the FRA start date
# def FRA(r1,r2,t1,t2,f=0):
#     #determine simple interest forward rates
#         if r2==0:
#             r2 = ((1.+r1*t1)*(1.+f*(t2-t1))-1)/t2
#             return r2
#         else:
#             f = ((1.+r2*t2)/(1+r1*t1)-1)/(t2-t1)#simple interest forward rates
#             return f
# FRA(0.035,0,(2/52.),(4/52.),0.0375)'''
# 
# 
# # In[1321]:
# 
# '''#Present Value of FRA
# #The appropriate discount rate is the the market rate for that tenor on the maturity date (Fixing)
# def pv_FRA(np,r_fra,t_fra,r_fix):
#     pv = np * (1.-(1.+r_fra * t_fra)/(1.+r_fix * t_fra))
#     return pv'''
# 
# 
# # In[1229]:
# 
# '''#Compound Interest –Discount Factors with only annual pmts
# def cmp_df(r, t):
#     df_lst = []
#     for i in range(1,t+1):
#         df = 1./(1.+(r))**(i)
#         df_lst.append(df)
#     return df_lst'''
# 
# 
# # In[1278]:
# 
# '''#YTM and Par Yield Curve
# #The par yield give a methodology whereby bonds of different coupon
# #but the same maturity can be compared
# #Pricing bullet bonds using the par yield methodology
# #Par yield (or par rate) denotes in finance, the coupon rate for which
# #the price of a bond is equal to its nominal value (or par value).
# #It is used in the design of fixed interest securities and in constructing interest rate swaps.
# #Deriving a par yield curve is a step toward creating a theoretical spot rate yield curve,
# #which is then used to more accurately price a coupon-paying bond. A method known as bootstrapping
# #is used to derive the arbitrage-free forward interest rates.
# def par_yield_pricing(c,r,t):
#     #with only annual pmts
#     cum_cp_value = np.zeros(t)
#     for i in range(1,t+1):
#         cp_value = c/(1.+r)**i
#         cum_cp_value[i-1] = cp_value
#     pv_cf = np.cumsum(cum_cp_value)
#     pv_bond = pv_cf[-1] + 100./(1.+r)**t
#     return pv_bond'''
# 
# 
# # In[ ]:
# 
# 
# 
# 
# # In[18]:
# 
# '''#NOTE: ORIGINAL, DO NOT CHANGE IT
# #Bootstrapping
# #these discount factors are effectively the prices of zero coupon bonds of that maturity
# 
# def boot_yield_curve_DF(yield_curve):
#     #perform an interpolation of the yield curve because longer maturities won't be in sequence
#     tm = np.array(yield_curve.keys())
#     yr = np.array(yield_curve.values())
#     interp = spi.interp1d(tm,yr,)
#     curve = {}
#     for i in range(1,tm.max()+1):
#         value = float(interp(i))
#         curve[i] = value
#     lista = []#empity list created to allocate the yield curve maturities
#     for maturities in curve:
#         lista.append(maturities)
#     df_lst = {}
#     for i in lista:
#         if i==1:
#             df1 = 1./(1.+curve[i])
#             df_lst[i] = df1
#         else:
#             suma=0
#             for j in range(1,i):
#                 suma = suma + (curve[i] * df_lst[j])
#             df_n = (1. - suma)/(1. + curve[i])
#             df_lst[i] = df_n
#     return df_lst'''
# 
# 
# # In[39]:
# 
# #function to generate the yield of a bond based on the cash flows
# #The bond yield is effectively the “IRR”of the bond
# def bond_yield(cash_flow):
#     byield = np.irr(cash_flow)
#     return byield
# 
# 
# # In[40]:
# 
# cf = [-104.19,8,8,8,8,8,8,8,8,8,108]
# bond_yield(cf)
# 
# 
# # In[1241]:
# 
# '''#Spot Yields
# def spot_yield(rates):
#     df = {}
#     for i in rates:
#         spot_n = (1.0/rates[i])**(1.0/i) - 1
#         df[i] = spot_n
#     return df'''
# 
# 
# # In[33]:
# 
# '''#pricing a bond using spot curve
# #Note: the spot curve is generated from the Discount Factor curve generated from the Par Yield Curve
# #and as a proof the bond value from with application of Spot Cruve should be the same if we employ
# #the Discount Factor curve
# def bond_pricing_spot_curve(cp,t,spot_curve):
#     pv_cp = 0
#     for i in range(1,t+1):
#         pv_cp+= cp/((1+spot_curve[i])**i)
#     pv_princ = 100/((1+spot_curve[t])**t)
#     price = pv_cp + pv_princ
#     return price
# 
# bond_pricing_spot_curve(6,3,s_spot)'''
# 
# 
# # In[1347]:
# 
# '''#princing the same bond with Discount Factor curve as proof
# #it should generate the same price a per Spot Curve
# def bond_pricing_DF(cp,t,df_curve):
#     pv_cp = 0
#     for i in range(1,t+1):
#         pv_cp+= cp * df_curve[i]
#     pv_princ = 100 * df_curve[t]
#     price = pv_cp + pv_princ
#     return price
# 
# bond_pricing_DF(6,3,bs_df2)'''
# 
# 
# # In[1305]:
# 
# '''#bumpping the spot curve up and down by 1bps
# spot_up = {}
# spot_down = {}
# for i in s_spot:
#     spot_up[i] = s_spot[i] + 0.0001
#     spot_down[i] = s_spot[i] - 0.0001'''
# 
# 
# # In[ ]:
# 
# 
# 
# 
# # In[ ]:
# 
# 
# 
# 
# # In[37]:
# 
# '''PV01 = -(p_up - p_down)/2'''
# 
# 
# # In[ ]:
# 
# 
# 
# 
# # In[ ]:
# 
# 
# 
# 
# # In[ ]:
# 
# 
# 
# 
# # In[ ]:
# 
# 
# 
# 
# # In[ ]:
# 
# 
# 
# 
# # In[ ]:
# 
# 
# 
# 
# # In[ ]:
# 


