import statsmodels.tsa.stattools as stt
import numpy as np

def SelectOrder(X,max_lags=10):
    filled = np.nan_to_num(X[:5000])
    pacf,confint = stt.pacf(filled,alpha=.05,nlags=max_lags)
    lags = np.where((confint[:,1]<0) | (0<confint[:,0]))[0]
    if(len(lags)<=1): 
        return np.array([1])
    else:
        return lags[1:]
