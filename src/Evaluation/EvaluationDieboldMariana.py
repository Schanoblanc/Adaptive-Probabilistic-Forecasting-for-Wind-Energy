import numpy as np
import scipy.stats as stats
from statsmodels.tsa.stattools import acovf

def Diebold_Mariana_Statistics(loss1:np.ndarray,loss2:np.ndarray,horizon=1):
    """
    Calculate the DM statistics
    l1 and l2 are the two loss series, e.g., CRPS or MSE, have same length; can contain NaN values.

    """
    assert len(loss1) == len(loss2), "should have same size"
    assert len(loss1.shape) == 1, "l1 should be 1D array"
    assert len(loss2.shape) == 1, "l2 should be 1D array"

    d_residual = loss1 - loss2
    n = np.count_nonzero(~np.isnan(d_residual))
    d_mean = np.nanmean(d_residual)
    nlag=horizon-1
    acovf_d = acovf(d_residual, missing="conservative", nlag=nlag)
    d_var = (acovf_d[0] + 2 * np.sum(acovf_d[1:nlag+1])) / n
    d_std = np.sqrt(d_var)
    if d_std < 1e-8: d_std = 1e-8
    dm_statistic = d_mean / d_std

    HLN_corrector = np.sqrt((n + 1 - 2*horizon + horizon * horizon**2 / n**2) / n)
    dm_statistic *= HLN_corrector
    return n, d_mean,d_std, dm_statistic

def Diebold_Mariana_Statistics_Matrix(losses:dict[str,np.ndarray],horizon=1):
    result = {}
    models = list(losses.keys())
    for i, model in enumerate(models):
        result[(model,model)] = (0.0,1.0,1.0,0.0) ## diagnoal element.
        for j in range(i+1, len(models)):
            model2 = models[j]
            loss1 = losses[model]
            loss2 = losses[model2]
            n, d_mean,d_std, dm_statistic = Diebold_Mariana_Statistics(loss1, loss2, horizon)
            result[(model, model2)] = (n, d_mean, d_std, dm_statistic)
            result[(model2, model)] = (n, -d_mean, d_std, -dm_statistic)  # symmetric on p value (for H0: T=0), anti symmetric on statistic
    return result