import numpy as np
import DataCleaning as DC
import scipy
from Models.IModel import IDBdSeqPtPredModel, IProbaPredResult, IDBdSeqPrbPredModel
from DataObjectModels.ProbPredResultDO import InflatedGaussianPred, NumeDoubleBoundProbaPred

class ARFlxLogitModel(IDBdSeqPtPredModel,IDBdSeqPrbPredModel):
    """
    AR Fix Logit + nu is obtimised in Fit. No adaptive update during forecasting.
    """
    lags: list[int] = [1]  # degree of model
    nu: float = None  # nu in logit transform
    param: np.ndarray = None
    sigma: float = None
    IsTrained: bool = False
    _verbose = False
    _Monitor=[]

    def __init__(self,lags, inital_nu=1.0, epsilon=0.005,resolution=1001, verbose=False):
        self.lags = lags
        self.nu = inital_nu
        self.param = np.zeros((len(lags)+1,1))
        self.sigma = 1
        self.__EPSILON = epsilon
        self.__RESOLUTION = resolution
        self._verbose = verbose

    def _partial_fit(self, X):
        LogitX = DC.Tranformation.LogitNormal(X, self.nu)
        Y_m = LogitX[:,np.newaxis]
        X_m = DC.Tranformation.SeqToOffsetLagMatrix(LogitX, self.lags)
        X_train, Y_train = DC.Tranformation.XYDropNaN(X_m, Y_m)
        if(len(X_train)==0): return # non valid training X Y pair

        self.param = np.linalg.solve(X_train.transpose() @ X_train, X_train.transpose() @ Y_train)
        data_Y_pred = X_train @ self.param
        residual = data_Y_pred - Y_train
        self.sigma = np.sqrt(np.nanmean(residual**2))
        return

    def _partial_fit_nu(self,X, nv_bounds:list[float]=[0.1,3]):
        res = scipy.optimize.minimize_scalar( lambda v: self.NegLogLikelihood(X,v),bounds=nv_bounds, options=dict(maxiter=200))
        self.nu = res.x
        return res

    def Fit(self, X, nv_bounds:list[float]=[0.1,3], max_iter = 1000, tolerance = 1E-20):
        pre_nll = self.NegLogLikelihood(X,self.nu)
        downcounter = max_iter
        nonimprv_downcounter = 10
        # print(f"fix initial, nu={self.nu:.5f}, mu={np.round(self.param.flatten(),3)}, sigma={self.sigma}, nll={self.NegLogLikelihood(X,self.nu):.5f}")
        # self._Monitor.append({"nu":np.copy(self.nu), "param":np.copy(self.param)})
        while(downcounter):
            self._partial_fit(X)
            res = self._partial_fit_nu(X,nv_bounds)
            imprv = res.fun - pre_nll
            pre_nll = res.fun

            # print(f"fix step {max_iter-downcounter+1}, nu={self.nu:.5f}, mu={np.round(self.param.flatten(),3)}, sigma={self.sigma}, nll={self.NegLogLikelihood(X,self.nu):.5f}")
            # self._Monitor.append({"nu":np.copy(self.nu), "param":np.copy(self.param)})
            if(imprv < tolerance): nonimprv_downcounter -= 1
            else: nonimprv_downcounter = 10
            
            if(nonimprv_downcounter == 0): break
            if(downcounter % 100 == 0 and self._verbose): print(f"iter-{max_iter-downcounter}, lost:{pre_nll}",end="\r")
            downcounter-= 1
        if(self._verbose):
                print(f"finished at iter-{max_iter-downcounter}, lost:{pre_nll}",end="\r")
        self.IsTrained = True

    def Reset(self, X, lags, nu):
        self.__init__(X, lags, nu)
        self.IsTrained = False
        self.nu = 0.5
        self.param = np.zeros((len(lags)+1,1))
        self.sigma = 1

    def NegLogLikelihood(self, X:np.array, nu=None):
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

    def PointPred(self, X):
        if(not self.IsTrained): Warning("Model not yet trained")
        LogitX = DC.Tranformation.LogitNormal(X, self.nu)
        X_m = DC.Tranformation.SeqToOffsetLagMatrix(LogitX, self.lags)
        res = X_m @ self.param
        origin_res = DC.Tranformation.InverseLogitNormal(res,self.nu)
        return origin_res

    def PointPredBounded(self, X, left, right):
        res = self.PointPred(X)
        res[res<left] = left
        res[res>right] = right
        return res
    
    def ProbaPred(self, X)-> IProbaPredResult: 
        if(not self.IsTrained): Warning("Model not yet trained")
        LogitZ = DC.Tranformation.LogitNormal(X, self.nu)
        Z_m = DC.Tranformation.SeqToOffsetLagMatrix(LogitZ, self.lags)
        trans_pred = (Z_m @ self.param).flatten()
        origin_res = [None] * len(X)

        trans_left = DC.Tranformation.LogitNormal(self.__EPSILON,self.nu)
        trans_right =DC.Tranformation.LogitNormal(1-self.__EPSILON,self.nu)
        for i in range((len(X))):
            x = trans_pred[i]
            if(np.isnan(x)): continue
            analytic_trans = InflatedGaussianPred(x,self.sigma,trans_left,trans_right) 
            origin_quentiles = np.linspace(self.__EPSILON,1-self.__EPSILON,self.__RESOLUTION)
            trans_quentiles = DC.Tranformation.LogitNormal(origin_quentiles,self.nu)
            origin_res[i] = NumeDoubleBoundProbaPred(origin_quentiles,analytic_trans.Cdfs(trans_quentiles)) 
        return origin_res