import numpy as np
import DataCleaning as DC
from Models.IModel import IDBdSeqPtPredModel

class ARFixModel(IDBdSeqPtPredModel):
    lags: list[int] = [1]  # degree of model
    param: np.ndarray = None
    sigma: float = None
    IsTrained: bool = False

    def __init__(self, lags):
        
        self.lags = lags
        self.param = np.zeros((len(lags)+1,1))
        self.sigma = 1

    def Fit(self, X):
        Y_m = X[:,np.newaxis]
        X_m = DC.Tranformation.SeqToOffsetLagMatrix(X, self.lags)
        X_train, Y_train = DC.Tranformation.XYDropNaN(X_m, Y_m)
        if(len(X_train)==0): return # non valid training X Y pair

        self.param = np.linalg.solve(X_train.transpose() @ X_train, X_train.transpose() @ Y_train)
        data_Y_pred = X_train @ self.param
        residual = data_Y_pred - Y_train
        assert(len(residual.shape)==1)
        self.sigma = np.sqrt(np.average(residual**2))

        self.IsTrained = True

    def Reset(self, X, lags, nu):
        self.__init__(X, lags, nu)
        self.IsTrained = False
        self.param = np.zeros((len(lags),1))
        self.sigma = 1

    def NegLogLikelihood(self, X):
        N = X.shape[0]
        termA1 = N * 0.5 * np.log(2*np.pi)
        termA2 = N * np.log(self.sigma)

        X_m = DC.Tranformation.SeqToOffsetLagMatrix(X, self.lags)
        pred = X_m @ self.param
        resd_term = 0.5 * (pred - self.LogitX[:,np.newaxis])**2 / self.sigma**2
        termC = np.nansum(resd_term)

        negloglh = termA1 + termA2 + termC
        return negloglh

    def PointPred(self, Z):
        if(not self.IsTrained): Warning("Model not yet trained")
        Z_m = DC.Tranformation.SeqToOffsetLagMatrix(Z, self.lags)
        res = Z_m @ self.param
        return res
    
    def PointPredBounded(self, Z, left, right):
        res = self.PointPred(Z)
        res[res<left] = left
        res[res>right] = right
        return res
