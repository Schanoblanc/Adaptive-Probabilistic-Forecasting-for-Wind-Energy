import DataObjectModels.ProbPredResultDO as ProbPredResultDO
import numpy as np
from typing import List

def PPEvaluation_TruncGaussians(Y:np.array,Pred_Y:np.array,sigmas,left_trunc, right_trunc, probas=np.linspace(0,1,101)):
    """
    Estimate the Accumulated Y-CDF value, against the Pred_Y quantiles.
    """
    assert len(Y) == len(Pred_Y), "should have same size"
    assert len(Y) == len(sigmas), "should have same size"
    if len(Pred_Y.shape) == 2: Pred_Y = Pred_Y.flatten()
    trcGaussian = ProbPredResultDO.TruncatedGaussianPred(0,1,left_trunc,right_trunc)
    accumul = np.zeros(len(probas))
    N = 1
    for idx in range(len(Y)):
        y = Y[idx]
        pred_y = Pred_Y[idx]
        sigma = sigmas[idx]
        if(np.isnan(y) or np.isnan(pred_y) or np.isnan(y) or np.isnan(pred_y)): continue
        if(y == pred_y and y == right_trunc): continue
        if(y == pred_y and y == left_trunc): continue
        N+=1
        trcGaussian.resetMoments(pred_y,sigma)
        quantiles = trcGaussian.Quantiles(probas)
        accumul +=  1*(y<quantiles)
    accumul /= N
    return accumul

def PPEvaluation_InflatedGaussians(Y,Pred_Y,sigmas,left_trunc, right_trunc, probas=np.linspace(0,1,101)):
    """
    Estimate the Accumulated Y-CDF value, against the Pred_Y quantiles.
    """
    assert len(Y) == len(Pred_Y), "should have same size"
    assert len(Y) == len(sigmas), "should have same size"
    if len(Pred_Y.shape) == 2: Pred_Y = Pred_Y.flatten()
    sqzGaussian = ProbPredResultDO.InflatedGaussianPred(0,1,left_trunc,right_trunc)
    accumul = np.zeros(len(probas))
    N = 1
    for idx in range(len(Y)):
        y = Y[idx]
        pred_y = Pred_Y[idx]
        sigma = sigmas[idx]
        if(np.isnan(y) or np.isnan(pred_y) or np.isnan(y) or np.isnan(pred_y)): continue
        if(y == pred_y and y == right_trunc): continue
        if(y == pred_y and y == left_trunc): continue
        N+=1
        sqzGaussian.resetMoments(pred_y,sigma)
        quantiles = sqzGaussian.Quantiles(probas)
        test = y<quantiles
        accumul +=  1*(y<quantiles)
    accumul /= N
    return accumul


def PPEvaluation_ProbaPred(Y,Pred_Y:List[ProbPredResultDO.IProbaPredResult], probas=np.linspace(0,1,101)):
    """
    Estimate the Accumulated Y-CDF value, against the Pred_Y quantiles.
    """
    assert len(Y) == len(Pred_Y), "should have same size"
    accumul = np.zeros(len(probas))
    N = 1
    for idx in range(len(Y)):
        y = Y[idx]
        pred_y = Pred_Y[idx]
        if(np.isnan(y) or pred_y == None): continue
        N+=1
        quantiles = pred_y.Quantiles(probas)
        # test = y<quantiles
        accumul +=  1*(y<quantiles)
    accumul /= N
    return accumul
    
