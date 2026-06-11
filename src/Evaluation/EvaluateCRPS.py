import numpy as np
import DataObjectModels.ProbPredResultDO as PPDO
import scipy.integrate as integ

__doc__="""
eCRPS: empirical CRPS. the fast one.
"""

def eCRPS(xs,ys,obs,cdf_obs):
    index_y = np.argmin(xs<obs)

    X_lhs = np.append(xs[0:index_y],obs)
    ys_lhs = np.append(ys[0:index_y],cdf_obs)
    Y_lhs = ys_lhs ** 2
    CRPS_lhs = integ.simpson(Y_lhs,X_lhs)

    X_rhs = np.append(obs, xs[index_y:])
    y_rhs = np.append(cdf_obs, ys[index_y:])
    Y_rhs = (1-y_rhs) ** 2
    CRPS_rhs = integ.simpson(Y_rhs,X_rhs)

    return CRPS_lhs + CRPS_rhs

eCRPSs = np.vectorize(eCRPS,signature="(n),(n),(),() -> ()")

def numeCRPS(ppred:PPDO.IProbaPredResult,obs:float,xmin:float, xmax:float,resolution=1001):
    xs = np.linspace(xmin,xmax,resolution)
    ys = ppred.Cdfs(xs)
    cdf_obs =ppred.Cdf(obs) 
    res = eCRPS(xs,ys,obs,cdf_obs)
    return res

numeCRPSs = np.vectorize(numeCRPS,excluded=['resolution'])

def NanNumeCRPS(ppred:PPDO.IProbaPredResult|None,obs:float,xmin:float, xmax:float,resolution=1001):
    if(ppred==None or obs==None or np.isnan(obs)): return np.nan
    return numeCRPS(ppred,obs,xmin,xmax,resolution)

NanNumeCRPSs = np.vectorize(NanNumeCRPS,excluded=['resolution'])

def CRPS(ppred:PPDO.IProbaPredResult,obs, xmin,xmax):
    """
    Use Integration to calculated CRPS: $\text{CRPS}(F(y|x),y) := \int_{-\infty}^{+\infty} (\mathcal{F}(y|x) - \mathbbm{1}_{y\leq x})^2 dx$
    
    Parameters:-----------------
    ppred: a Probabilistic Prediction, which the Cdf(quantile) and Cdfs([quantiles]) are callable to get cdf(s).
    obs: float, the true observation
    xmin: the min of possible observation, thus the lower bound of integration.
    xmax: the max of ~, thus the upper bound of ~.

    Returns: CRPS value, estimation of error

    Reference:------------------
    Estimation of the Continuous Ranked Probability Score with Limited Information and Applications to Ensemble Weather Forecasts, Michaël Zamo & Philippe Naveau 
    https://doi.org/10.1007/s11004-017-9709-7
    """
    def lft(x): return ppred.Cdf(x) **2
    def rht(x): return (1-ppred.Cdf(x))**2

    if(obs<=xmin): return integ.quad(rht,obs,xmax)
    if(xmax<=obs): return integ.quad(lft,xmin,obs)
    lhs, errorl = integ.quad(lft,xmin,obs)
    rhs,errorr = integ.quad(rht,obs,xmax)
    return lhs + rhs, max(errorl,errorr)

CRPSs = np.vectorize(CRPS)
