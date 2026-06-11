import numpy as np
import DataCleaning as DC
from Models.IModel import IDBdSeqPtPredModel, IProbaPredResult, IDBdSeqPrbPredModel
from DataObjectModels.ProbPredResultDO import InflatedGaussianPred, NumeDoubleBoundProbaPred

class ARFixLogitModel(IDBdSeqPtPredModel,IDBdSeqPrbPredModel):
    """
    Notice: param and sigma is in transformed domain. 
            When forecast, forecast a proba dist at transformed doamin,
            Then, transform back the quantile CDF to original doamin.
    """
    lags: list[int] = [1]  # degree of model
    nu: float = 0.5  # nu in logit transform
    param: np.ndarray = None
    sigma: float = None
    IsTrained: bool = False

    def __init__(self, lags, nu,epsilon=0.005,resolution=1001):
        self.lags = lags
        self.nu = nu
        self.__EPSILON = epsilon
        self.__RESOLUTION = resolution
    
        self.param = np.zeros((len(lags)+1,1))
        self.sigma = 1

    def Fit(self,X):
        LogitX = DC.Tranformation.LogitNormal(X, self.nu)
        Y_m = LogitX[:,np.newaxis]
        X_m = DC.Tranformation.SeqToOffsetLagMatrix(LogitX, self.lags)
        X_train, Y_train = DC.Tranformation.XYDropNaN(X_m, Y_m)
        if(len(X_train)==0): return # non valid training X Y pair
        
        self.param = np.linalg.solve(X_train.transpose() @ X_train, X_train.transpose() @ Y_train)
        data_Y_pred = X_train @ self.param
        residual = data_Y_pred - Y_train
        self.sigma = np.sqrt(np.average(residual**2))

        self.IsTrained = True
        return

    def Reset(self, X, lags, nu):
        self.__init__(X, lags, nu)
        self.IsTrained = False
        self.param = np.zeros((len(lags)+1,1))
        self.sigma = 1

    def NegLogLikelihood(self, X, nu=None):
        if(not nu): nu = self.nu
        LogitX = DC.Tranformation.LogitNormal(X, nu)
        Y_trans = LogitX[:,np.newaxis]
        Y_origin = X[:,np.newaxis]
        X_m = DC.Tranformation.SeqToOffsetLagMatrix(LogitX, self.lags)
        _, Y = DC.Tranformation.XYDropNaN(X_m, Y_origin)

        N = Y.shape[0]
        termA1 = N * 0.5 * np.log(2*np.pi)
        termA2 = N * np.log(self.sigma)
        
        termB1 = -1 * N * np.log(nu)
        Ynu = Y**nu
        termB2 = np.nansum(np.log(Y*(1 - Ynu)))

        pred = (X_m @ self.param)
        resd_term = 0.5 * (pred - Y_trans)**2 / self.sigma**2
        termC = np.nansum(resd_term)

        negloglh = termA1 + termA2 + termB1 + termB2 + termC
        # print(f"NLL={negloglh:.5f} A={termA1 + termA2:.5f}, B={termB1+termB2:.5f}, C={termC:.5f}")
        return negloglh

    def PointPred(self, Z):
        if(not self.IsTrained): Warning("Model not yet trained")
        LogitZ = DC.Tranformation.LogitNormal(Z, self.nu)
        Z_m = DC.Tranformation.SeqToOffsetLagMatrix(LogitZ, self.lags)
        res = Z_m @ self.param
        origin_res = DC.Tranformation.InverseLogitNormal(res,self.nu).flatten()
        return origin_res
    
    def PointPredBounded(self, Z, left, right):
        res = self.PointPred(Z)
        res[res<left] = left
        res[res>right] = right
        return res
    
    def ProbaPred(self, X)-> IProbaPredResult: 
        # ======> Transformed Domain 

        if(not self.IsTrained): Warning("Model not yet trained")
        LogitZ = DC.Tranformation.LogitNormal(X, self.nu)
        Z_m = DC.Tranformation.SeqToOffsetLagMatrix(LogitZ, self.lags)
        trans_pred = (Z_m @ self.param).flatten()
        nume_res = [None] * len(X)

        trans_left = DC.Tranformation.LogitNormal(self.__EPSILON,self.nu)
        trans_right =DC.Tranformation.LogitNormal(1-self.__EPSILON,self.nu)
        for i in range((len(X))):
            x = trans_pred[i]
            if(np.isnan(x)): continue
            analytic_trans = InflatedGaussianPred(x,self.sigma,trans_left,trans_right) 
            origin_quentiles = np.linspace(self.__EPSILON,1-self.__EPSILON,self.__RESOLUTION)
            trans_quentiles = DC.Tranformation.LogitNormal(origin_quentiles,self.nu)
            nume_res[i] = NumeDoubleBoundProbaPred(origin_quentiles,analytic_trans.Cdfs(trans_quentiles)) 
        return nume_res