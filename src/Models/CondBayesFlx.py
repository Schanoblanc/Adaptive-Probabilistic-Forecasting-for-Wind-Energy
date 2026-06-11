import numpy as np
import scipy
import DataCleaning as DC
from DataObjectModels import ProbPredResultDO
from Models.IModel import IDBdSeqPrbPredModel, IProbaPredResult
from typing import List
import logging

__version = "2024.4.22"

class CondBayesFlxLogitModel(IDBdSeqPrbPredModel):
    """
    nu is flex. only params updated.
    """
    mu: np.ndarray
    Precision: np.ndarray
    L: np.ndarray # The cholesky decomposition of Precision matrix
    alpha: float = None
    beta: float = None
    nu: float = 0.5  # nu in logit transform

    lags: list[int] = [1]  # degree of model
    IsTrained: bool = False
    _nus:np.ndarray

    @property
    def PrecisionZ(self): return (self.alpha - 1)/self.beta

    def __init__(self, lags,mu,precision, alpha=11,beta=.1,nu=1, epsilon=0.005, resolution=1001,bounds=(0.1,3)):
        self.lags = lags
        assert(len(mu.shape)==1), "mu should be a 1D vector"
        self.mu = mu
        self.Precision = precision
        self.L = np.linalg.cholesky(precision) + epsilon
        self.alpha = alpha
        self.beta = beta
        self.nu = nu
        self.__EPSILON = epsilon
        self.__RESOLUTION = resolution
        self.nu_bound = bounds
        self.conti_incr = 0
        self.conti_decr = 0

    def Fit(self, X):
        ### For bayesian Parame
        LogitX = DC.Tranformation.LogitNormal(X, self.nu)
        Y_m = LogitX[:,np.newaxis]
        X_m = DC.Tranformation.SeqToOffsetLagMatrix(LogitX, self.lags) #shape(N,d)
        X_train, Y_train = DC.Tranformation.XYDropNaN(X_m, Y_m) #shape(N,d), (N,1)
        # print(X_train, Y_train)
        if(len(X_train)==0): return # non valid training X Y pair

        newPrecision = X_train.T @ X_train * self.PrecisionZ + self.Precision
        newMu = np.linalg.solve(newPrecision, X_train.T @ Y_train * self.PrecisionZ +  self.Precision @ self.mu[:,np.newaxis]).flatten()
        newAlpha = self.alpha + len(X_train) / 2
        newBeta = self.beta + ( Y_train.T @ Y_train + self.mu.T @ self.Precision @ self.mu / self.PrecisionZ  - newMu.T @ newPrecision @ newMu / self.PrecisionZ )[0,0] /2

        self.Precision = newPrecision
        self.mu = newMu
        self.alpha = newAlpha
        self.beta = newBeta
        self.L = np.linalg.cholesky(newPrecision)

        ### for nu
        # x0 = self.nu * (1+ (np.random.random()-0.5)/20)
        res = scipy.optimize.minimize(
            lambda nu_: self.NegLogLikelihood(X,nu_[0])+self.NegLogLikelihood_vari(nu_[0]),
            x0=self.nu,method='Nelder-Mead', bounds=[self.nu_bound], options=dict(maxiter=20))
        new_nu =  res.x[0]
        if(new_nu>self.nu): 
            self.conti_incr =min(self.conti_incr+1, 5)
            self.conti_decr = 1
        if(new_nu<self.nu): 
            self.conti_incr = 1
            self.conti_decr = min(self.conti_decr+1, 5)
        lr = 0.05 ** max(self.conti_incr,self.conti_decr)
        self.nu = (1-lr)*self.nu + lr * new_nu
        # res = scipy.optimize.minimize_scalar(lambda nu_: self.NegLogLikelihood_vari(X, nu_), bounds=self.nu_bound, options=dict(maxiter=20))
        # self.nu = res.x

        return
    
    def _propagonde(self):
        ### First Model Propagation ###
        self.Precision *= 0.995 #.999
        self.alpha *= 0.95 #.995
        self.beta *= 0.95 #.995
        # may add boundeary check later if needed

    def ProbaPred(self, X:np.ndarray)-> List[IProbaPredResult]: 

        ### Result Buffer ###
        NumData = len(X)
        result = [None] * NumData

        ### Logger
        self._nus = [None] * NumData

        ### input transformation ###
        LogitX = DC.Tranformation.LogitNormal(X, self.nu)
        X_m = DC.Tranformation.SeqToOffsetLagMatrix(LogitX, self.lags) #shape(N,d)
        fragment_len = max(self.lags)

        ### each by each prediction ###
        for row in range(NumData):
            # print(row)
            Xrow = X_m[row,:] 
            predYmu = (Xrow.T @ self.mu).flatten()[0] ### use [0] because mu is a one column matrix in this model
            varModel = Xrow.T @ np.linalg.solve(self.Precision,Xrow) ## do not include uncertainty introduced in this step to test
            varNoise = 1/self.PrecisionZ
            predYsigma = np.sqrt( varModel + varNoise) 
            # print(predYmu,predYsigma, varModel, 1/self.PrecisionZ, end="\r")

        ### Output back transformation + Update ###
            if(not np.isnan(predYmu) and not np.isnan(predYsigma)): 
                left = DC.Tranformation.LogitNormal(self.__EPSILON, self.nu)
                right = DC.Tranformation.LogitNormal(1-self.__EPSILON, self.nu)
                trans_pred = ProbPredResultDO.InflatedGaussianPred(mean=predYmu, sigma=predYsigma,left_trunc=left, right_trunc=right)
                origin_quentiles = np.linspace(self.__EPSILON,1-self.__EPSILON,self.__RESOLUTION)
                trans_quentiles = DC.Tranformation.LogitNormal(origin_quentiles,self.nu)
                # print(predYmu,predYsigma,self.nu,left, right, end="\r") # DEUBG
                result[row]= ProbPredResultDO.NumeDoubleBoundProbaPred(origin_quentiles,trans_pred.Cdfs(trans_quentiles))  
                self._propagonde()
                # print(X[row-fragment_len:row+1])
                self.Fit(X[row-fragment_len:row+1])

            self._nus[row] = self.nu
        return result
    
    def NegLogLikelihood(self,X,nu=None):
        if(not nu): nu = self.nu
        LogitX = DC.Tranformation.LogitNormal(X, nu)
        Y_trans = LogitX
        Y_origin = X[:,np.newaxis]
        X_m = DC.Tranformation.SeqToOffsetLagMatrix(LogitX, self.lags)
        _, Y = DC.Tranformation.XYDropNaN(X_m, Y_origin)

        N = Y.shape[0]
        sigma = np.sqrt(1/self.PrecisionZ)
        termA1 = N * 0.5 * np.log(2*np.pi)
        termA2 = N * np.log(sigma)
        
        termB1 = -1 * N * np.log(nu)
        Ynu = Y**nu
        termB2 = np.nansum(np.log(Y*(1 - Ynu)))

        pred = (X_m @ self.mu) # shape (N,)
        resd_term = 0.5 * (pred - Y_trans)**2 / sigma**2
        assert resd_term.shape == Y_trans.shape
        termC = np.nansum(resd_term)

        negloglh = termA1 + termA2 + termB1 + termB2 + termC
        # print(f"NLL={negloglh:.5f} A={termA1 + termA2:.5f}, B={termB1+termB2:.5f}, C={termC:.5f}")
        return negloglh
    
    def NegLogLikelihood_vari(self, nu=None):
        if(not nu): nu = self.nu

        ModelX_trans = self.L.T
        ### Rescale to avoid numerical issue
        max_X = np.max(np.abs(ModelX_trans))
        if(max_X > 5): ## set 1 because ingeneral the transdomain has 95% in [-10,10]
            normalisor = 5 / max_X
            ModelX_trans *= normalisor 
        TotalX_trans = ModelX_trans #np.vstack((ModelX_trans,X_m))

        ModelY_trans = (ModelX_trans @ self.mu) # shape(N,)
        ModelY = DC.Tranformation.InverseLogitNormal(ModelY_trans,self.nu) # shape(d,)
        ModelY_trans_new = DC.Tranformation.LogitNormal(ModelY,nu) # shape(d,)
        TotalY_trans = ModelY_trans_new #ModelY_trans #np.hstack((ModelY_trans_new,Y_trans)) # shape(N+d,)

        TotalY_effect =  ModelY #np.hstack((ModelY,Y.flatten())) 
        ### Change to longdouble to avoid overflow runtime issue

        N = TotalY_effect.shape[0]
        sigma = np.sqrt(1/self.PrecisionZ)
        termA1 = N * 0.5 * np.log(2*np.pi)
        termA2 = N * np.log(sigma)
        
        termB1 = -1 * N * np.log(nu)
        Ynu = TotalY_effect**nu
        b2s = np.log(TotalY_effect*(1 - Ynu)) # Should keep TotalY_effect for num stability, even it is just a const
        # b2s = np.log((1 - Ynu))
        termB2 = np.nansum(b2s)

        pred = (TotalX_trans @ self.mu) # shape (N,)
        residuals = pred - TotalY_trans
        resd_term = 0.5 * (residuals)**2 / sigma**2
        assert resd_term.shape == TotalY_trans.shape, "wrong prediction shape"
        termC = np.nansum(resd_term)

        negloglh = termA1 + termA2 + termB1 + termB2 + termC
        # print(f"nu on {nu:.3f} NLL={negloglh:.5f} A={termA1 + termA2:.5f}, B={termB1+termB2:.5f}, C={termC:.5f}")
        return negloglh
