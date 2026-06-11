import numpy as np
import DataCleaning as DC
from DataObjectModels import ProbPredResultDO
from Models.IModel import IDBdSeqPrbPredModel, IProbaPredResult
from typing import List

__version = "2024.4.22"

class BayesFixLogitModel(IDBdSeqPrbPredModel):
    """
    nu is fixed. only params updated.
    """
    mu: np.ndarray
    Precision: np.ndarray
    alpha: float = None
    beta: float = None
    nu: float = 0.5  # nu in logit transform

    lags: list[int] = [1]  # degree of model
    IsTrained: bool = False

    @property
    def PrecisionZ(self): return (self.alpha - 1)/self.beta

    def __init__(self, lags,mu,precision, nu, _forget, alpha=101,beta=1, epsilon=0.005, resolution=1001):
        ### Input Validation
        assert len(mu.shape)==1, "Mu should be 1D vector"

        self.lags = lags
        self.mu = mu
        self.Precision = precision
        self.alpha = alpha
        self.beta = beta
        self.nu = nu
        self._forget = _forget
        self.__EPSILON = epsilon
        self.__RESOLUTION = resolution

    def Fit(self, X):
        LogitX = DC.Tranformation.LogitNormal(X, self.nu)
        Y_m = LogitX[:,np.newaxis]
        X_m = DC.Tranformation.SeqToOffsetLagMatrix(LogitX, self.lags) #shape(N,d)
        X_train, Y_train = DC.Tranformation.XYDropNaN(X_m, Y_m) #shape(N,d), (N,1)
        if(len(X_train)==0): return # non valid training X Y pair
        # print(X_train, Y_train)

        newPrecision = X_train.T @ X_train * self.PrecisionZ + self.Precision
        newMu = np.linalg.solve(newPrecision, X_train.T @ Y_train * self.PrecisionZ +  self.Precision @ self.mu[:,np.newaxis]).flatten()
        newAlpha = self.alpha + len(X_train) / 2
        newBeta = self.beta + ( Y_train.T @ Y_train + self.mu.T @ self.Precision @ self.mu / self.PrecisionZ  - newMu.T @ newPrecision @ newMu / self.PrecisionZ )[0,0] /2

        self.Precision = newPrecision
        self.mu = newMu
        self.alpha = newAlpha
        self.beta = newBeta
        return
    
    def _propagonde(self):
        ### decay in sense for a exponetial smoothing
        self.Precision *= self._forget

        ### Handle numerical issue
        if(np.linalg.det(self.Precision)> 1E20): self.Precision *= 0.99 + 0.01*np.eye(len(self.mu))

        if(self.alpha<1E20 and self.beta<1E20): pass  # Do nonthing, no need to worry numerical issue
        else:
            self.alpha *= 0.995
            self.beta *= 0.995

        ### may add boundeary check later if needed
        return

    def ProbaPred(self, X:np.ndarray)-> List[IProbaPredResult]: 

        ### Result Buffer ###
        NumData = len(X)
        result = [None] * NumData

        ### input transformation ###
        LogitX = DC.Tranformation.LogitNormal(X, self.nu)
        X_m = DC.Tranformation.SeqToOffsetLagMatrix(LogitX, self.lags) #shape(N,d)
        fragment_len = max(self.lags)

        ### each by each prediction ###
        for row in range(NumData):
            Xrow = X_m[row,:] ### Shape changed to (d,) due to only 1 dim np array!
            predYmu = (Xrow.T@ self.mu).flatten()[0] # use [0] because mu is a one column matrix in this model
            varModel = Xrow.T@ np.linalg.solve(self.Precision,Xrow)
            varNoise = 1/self.PrecisionZ

            ### Output back transformation + Update ###
            if(np.isnan(predYmu) or np.isnan(varModel) or np.isnan(varNoise)):  continue
            ## else
            predYsigma = np.sqrt( varModel + varNoise) 
            # print(predYmu,predYsigma, varModel, 1/self.PrecisionZ, end="\r")
            left = DC.Tranformation.LogitNormal(self.__EPSILON, self.nu)
            right = DC.Tranformation.LogitNormal(1-self.__EPSILON, self.nu)
            trans_pred = ProbPredResultDO.InflatedGaussianPred(mean=predYmu, sigma=predYsigma,left_trunc=left, right_trunc=right)
            origin_quentiles = np.linspace(self.__EPSILON,1-self.__EPSILON,self.__RESOLUTION)
            trans_quentiles = DC.Tranformation.LogitNormal(origin_quentiles,self.nu)
            # print(predYmu,predYsigma,self.nu,left, right, end="\r") # DEUBG
            result[row]= ProbPredResultDO.NumeDoubleBoundProbaPred(origin_quentiles,trans_pred.Cdfs(trans_quentiles))
            self._propagonde()
            self.Fit(X[row-fragment_len:row+1])
        return result
    
