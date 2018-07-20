# Bonds priced using discount factors derived from Par Yield Curve

from scipy import interpolate 
import matplotlib.pyplot as plt

def main():
    (df) = runInterpolation()
    runPricing(df)

def calculateBondPrice(coupon, t, df, bp):
    coupon = coupon + bp
    price = 0
    face = 100
    for i in range(0,t-1):
        price += coupon * df[i]
    price += (face + coupon) * df[i+1]
    return(price)

def priceHalfYear(coupon, t, df, bp):
    c = [0.5, 1, 1, 101]
    half_year_df = []
    a = 1 / (1 + 0.5 * 1.12 / 100)
    half_year_df.append(a)
    for i in range(0, 3):
        a = (df[i+1] + df[i]) / 2
        half_year_df.append(a)
    price = 0
    for i in range(0,4):
        if i == 0 :
            price += (c[i] + bp/2) * half_year_df[i] ; continue
        price += (c[i] + bp) * half_year_df[i]
    return(price)

def calculateDiscountFactor(s, r):
    a = 100 - r
    b = 100 + s * r
    return(a/b)

def discountCurve(x,y):
    df = []
    par_at_one = y[0]
    factor = 1 / (1 + (1 * par_at_one) / 100)
    df.append(factor)
    for i in range(2, 31):
        df.append(calculateDiscountFactor(factor, y[i-1]))
        factor += df[i-1]    
    plt.plot(x, df)
    plt.show()
    return(df)

def runInterpolation():
    x = [0.0833, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]
    y = [0.89, 1, 1.12, 1.22, 1.38, 1.51, 1.79, 2.02, 2.21, 2.6, 2.87]
    z = range(1,31)
    f = interpolate.interp1d(x,y)
    i = f(z)
    df = discountCurve(z, i)
    print(df)
    return(df)

def runPricing(df):
    print("")
    print("3.5 Year bond paying 1% coupon")
    price1 = priceHalfYear(1, 3.5, df, 0)
    print("Bond Price: %f" % price1)
    price2 = priceHalfYear(1, 3.5, df, 0.01)
    dv01 = abs(price1 - price2) ; k = dv01
    print("DVO1: %f" % dv01)      
    print("")
    
    print("15 Year bond paying 4% coupon")    
    price1 = calculateBondPrice(4, 15, df, 0)
    print("Bond Price: %f" % price1)
    price2 = calculateBondPrice(4, 15, df, 0.01)
    dv01 = abs(price1 - price2)    
    print("DVO1: %f" % dv01)    
    i = 100000000 / price1
    j = i*dv01
    print("")
     
    print("25 Year bond paying 2% coupon")    
    price1 = calculateBondPrice(2, 25, df, 0)
    print("Bond Price: %f" % price1)
    price2 = calculateBondPrice(2, 25, df, 0.01)
    dv01 = abs(price1 - price2) ; l = dv01
    print("DVO1: %f" % dv01)        
    print("")
 
    print("If holding 100M worth of 15 year, monetary value of DV01: $%.2f" % j)
    print("To have an equivalent level of risk when holding the 3.5 year, you would need to purchase %d bonds" % (j/k))
    print("To have an equivalent level of risk when holding the 25 year, you would need to purchase %d bonds" % (j/l))      

main()