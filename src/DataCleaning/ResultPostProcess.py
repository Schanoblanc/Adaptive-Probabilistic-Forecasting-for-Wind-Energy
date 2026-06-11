from DataObjectModels import ProbPredResultDO as PPDO
import numpy as np
from typing import Iterable, List

def PairFilterOutEmpty(A:Iterable, B:Iterable):
    OK_A = []
    OK_B = []
    for a, b in zip(A,B):
        if(a != None and ~np.isnan(b)):
            OK_A.append(a)
            OK_B.append(b)
    return OK_A,OK_B

def PairFilterOutNan(A:np.array, B:np.array):
    index = ~np.isnan(A) & ~np.isnan(B)
    return A[index],B[index]

def GetQuantilePred(PredY:List[PPDO.IProbaPredResult],at_cdf):
    NumData = len(PredY)
    result = [None] * NumData
    for ind, pred in enumerate(PredY):
        if(pred != None): result[ind] = pred.Quantile(at_cdf)
    return result