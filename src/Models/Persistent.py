import numpy as np
import DataCleaning as DC
from Models.IModel import IDBdSeqPtPredModel, IDBdSeqPrbPredModel, IProbaPredResult
from DataObjectModels.ProbPredResultDO import InflatedGaussianPred, NumeDoubleBoundProbaPred

class PersistentModel(IDBdSeqPtPredModel,IDBdSeqPrbPredModel):
    param: np.ndarray = None
    sigma: float = None
    IsTrained: bool = False

    def __init__(self,epsilon=0.005,resolution=1001):
        self.sigma = 1
        self.__EPSILON = epsilon
        self.__RESOLUTION = resolution

    def Fit(self,X):
        Y_m = X[:,np.newaxis]
        X_m = DC.Tranformation.SeqToOffsetLagMatrix(X, [1])
        X_train, Y_train = DC.Tranformation.XYDropNaN(X_m, Y_m)
        if(len(X_train)==0): return # non valid training X Y pair

        data_Y_pred = X_train[:,1]
        residual = data_Y_pred.flatten() - Y_train.flatten()
        self.sigma = np.sqrt(np.average(residual**2))

        self.IsTrained = True

    def Reset(self, lags, nu):
        self.__init__()
        self.IsTrained = False
        self.param = np.zeros((len(lags),1))
        self.sigma = 1

    def NegLogLikelihood(self,X):
        N = X.shape[0]
        termA1 = N * 0.5 * np.log(2*np.pi)
        termA2 = N * np.log(self.sigma)

        pred = self.PointPred(X)
        resd_term = 0.5 * (pred - X[:,np.newaxis])**2 / self.sigma**2
        termC = np.nansum(resd_term)

        negloglh = termA1 + termA2 + termC
        return negloglh

    def PointPred(self, Z):
        if(not self.IsTrained): Warning("Model not yet trained")
        Z_m = DC.Tranformation.SeqToOffsetLagMatrix(Z, [1])
        res = Z_m[:,1]
        return res
    
    def PointPredBounded(self, Z, left, right):
        res = self.PointPred(Z)
        res[res<left] = left
        res[res>right] = right
        return res
    
    def ProbaPred(self, X)-> IProbaPredResult: 
        left = self.__EPSILON
        right = 1-self.__EPSILON
        PredX = self.PointPredBounded(X,left,right)
        nume_res = [None] * len(X)
        origin_quentiles = np.linspace(self.__EPSILON,1-self.__EPSILON,self.__RESOLUTION)
        for i in range((len(X))):
            x = PredX[i]
            if(np.isnan(x)): continue
            analytic_res = InflatedGaussianPred(x,self.sigma,left,right) 
            trans_quentiles = origin_quentiles
            nume_res[i] = NumeDoubleBoundProbaPred(origin_quentiles,analytic_res.Cdfs(trans_quentiles)) 
        return nume_res
