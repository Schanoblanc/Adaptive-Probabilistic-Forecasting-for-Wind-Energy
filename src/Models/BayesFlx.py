import numpy as np
import scipy
import DataCleaning as DC
from DataObjectModels import ProbPredResultDO
from Models.IModel import IDBdSeqPrbPredModel, IProbaPredResult
from typing import List
import logging

__version = "2024.4.22"

class BayesFlxLogitModel(IDBdSeqPrbPredModel):
    """
    nu is flex. only params updated.
    """
    mu: np.ndarray
    p:int #order
    Precision: np.ndarray
    L: np.ndarray # The cholesky decomposition of Precision matrix
    alpha: float = None
    beta: float = None
    nu: float = 0.5  # nu in logit transform

    lags: list[int] = [1]  # degree of model
    IsTrained: bool = False
    _nus:np.ndarray
    _max_eigval:np.ndarray
    _min_eigval:np.ndarray
    _precision:np.ndarray
    _L_update_succes : bool

    @property
    def PrecisionZ(self): return max((self.alpha - 1)/self.beta, 1E-8)

    def __init__(self, lags,mu,precision, nu, _forget, alpha=11,beta=.1, epsilon=0.005, resolution=1001,bounds=(0.1,3)):
        self.lags = lags
        assert(len(mu.shape)==1), "mu should be a 1D vector"
        self.mu = mu
        self.p = len(mu)
        self.Precision = precision
        self.L = np.linalg.cholesky(precision) + epsilon
        self.alpha = alpha
        self.beta = beta
        self.nu = nu
        self._forget = _forget
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
        try:
            newL = np.linalg.cholesky(newPrecision)
            self.L = newL
            self._L_update_succes = True
        except np.linalg.LinAlgError:
            self._L_update_succes = False
            # print(X_train,self.PrecisionZ, self.Precision, "update failed")
            return

        ## Only Update if Precision is in good condition
        newMu = np.linalg.solve(newPrecision, X_train.T @ Y_train * self.PrecisionZ +  self.Precision @ self.mu[:,np.newaxis]).flatten()
        newAlpha = self.alpha + len(X_train) / 2
        newBeta = self.beta + ( Y_train.T @ Y_train + self.mu.T @ self.Precision @ self.mu / self.PrecisionZ  - newMu.T @ newPrecision @ newMu / self.PrecisionZ )[0,0] /2

        self.Precision = newPrecision
        self.mu = newMu
        self.alpha = newAlpha
        self.beta = newBeta

        ### for nu
        ## x0 = self.nu * (1+ (np.random.random()-0.5)/20)
        res = scipy.optimize.minimize(
            lambda nu_: self.NegLogLikelihood(X,nu_[0])+self.NegLogLikelihood_vari2(nu_[0]),
            x0=self.nu,method='L-BFGS-B', bounds=[self.nu_bound], options=dict(maxiter=20))
        # res = scipy.optimize.minimize(
        #     lambda nu_: self.NegLogLikelihood_vari(nu_[0]),
        #     x0=self.nu,method='L-BFGS-B', bounds=[self.nu_bound], options=dict(maxiter=20))
        new_nu =  res.x[0]
        # print(f"({self.nu},{self.NegLogLikelihood_vari()})\n-->{new_nu,self.NegLogLikelihood_vari(new_nu)}")

        lr = 0.05
        self.nu = (1-lr)*self.nu + lr * new_nu

        return
    
    def _propagonde(self):
        if(not self._L_update_succes): 
            self.Precision = self.Precision*0.99 + 0.01*np.eye(len(self.mu)) #.999
            return
        ### First Model Propagation ###
        self.Precision *= self._forget

        if(self.alpha<1E20 and self.beta<1E20): pass  # Do nonthing, no need to worry numerical issue
        else:
            self.alpha *= 0.9995
            self.beta *= 0.9995
        # may add boundeary check later if needed
        return

    def ProbaPred(self, X:np.ndarray)-> List[IProbaPredResult]: 

        ### Result Buffer ###
        NumData = len(X)
        result = [None] * NumData

        ### Logger
        self._nus = [self.nu] * NumData

        ### each by each prediction ###
        fragment_len = max(self.lags)
        for row in range(NumData):
            # print(row)
            if(row ==8787):
                hit=True

            ### input transformation ###
            if(row<= fragment_len): continue ## non computation for the starting
            x_predictor_y = X[row-fragment_len:row+1]
            LogitX = DC.Tranformation.LogitNormal(x_predictor_y, self.nu)
            X_m = DC.Tranformation.SeqToOffsetLagMatrix(LogitX, self.lags) #shape(N,d)
            Xrow = X_m[-1,:] 
            predYmu = (Xrow.T @ self.mu).flatten()[0] ### use [0] because mu is a one column matrix in this model

            rank = np.linalg.matrix_rank(self.Precision)
            if rank < self.Precision.shape[0]:
                logging.warning(f"Precision matrix rank {rank} < {self.Precision.shape[0]} at row {row}, using pseudo inverse")
                self.Precision = np.linalg.pinv(self.Precision)
                varModel = Xrow.T @ np.linalg.pinv(self.Precision) @ Xrow
            else:
                varModel = Xrow.T @ np.linalg.solve(self.Precision,Xrow)
            varNoise = 1/self.PrecisionZ
            predYsigma = np.sqrt(varModel + varNoise) 
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
                self.Fit(x_predictor_y)
                if(not self._L_update_succes):print(f"L update fail at data index {row}")

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
    
    def NegLogLikelihood_vari2(self, nu=None):
        if(not nu): nu = self.nu

        ModelX_trans = self.L.T
        ### Rescale to avoid numerical issue
        max_X = np.max(np.abs(ModelX_trans))
        normalisor = max_X
        thresd = np.max([DC.Tranformation.LogitNormal(1-2*self.__EPSILON,self.nu),-DC.Tranformation.LogitNormal(2*self.__EPSILON,self.nu)])
        if(max_X > thresd): ## set 1 because ingeneral the transdomain has 95% in [-10,10]
            normalisor = thresd / max_X
            ModelX_trans *= normalisor 
        TotalX_trans = ModelX_trans #np.vstack((ModelX_trans,X_m))

        ModelY_trans = (ModelX_trans @ self.mu) # shape(N,)
        ModelY = DC.Tranformation.InverseLogitNormal(ModelY_trans,self.nu) # shape(d,)
        ModelY_trans_new = DC.Tranformation.LogitNormal(ModelY,nu) # shape(d,)
        TotalY_trans = ModelY_trans_new #ModelY_trans #np.hstack((ModelY_trans_new,Y_trans)) # shape(N+d,)

        sigma = np.sqrt(1/self.PrecisionZ)
        pred = (TotalX_trans @ self.mu) # shape (N,)
        residuals = pred - TotalY_trans
        resd_term = 0.5 * (residuals)**2 / sigma**2
        assert resd_term.shape == TotalY_trans.shape, "wrong prediction shape"
        termC = np.nansum(resd_term)

        negloglh = termC
        # print(f"nu on {nu:.3f} NLL={negloglh:.5f} A={termA1 + termA2:.5f}, B={termB1+termB2:.5f}, C={termC:.5f}")
        return negloglh
    
    def NegLogLikelihood_vari(self, nu=None):
        raise ValueError("should not use this method becuase nv shifting issue due to fact that \
                         reconstructed data not get optimal at current nu")
        if(not nu): nu = self.nu

        ModelX_trans = self.L.T
        ### Rescale to avoid numerical issue
        max_X = np.max(np.abs(ModelX_trans))
        normalisor = max_X
        thresd = np.max([DC.Tranformation.LogitNormal(1-2*self.__EPSILON,self.nu),-DC.Tranformation.LogitNormal(2*self.__EPSILON,self.nu)])
        if(max_X > thresd): ## set 1 because ingeneral the transdomain has 95% in [-10,10]
            normalisor = thresd / max_X
            ModelX_trans *= normalisor 
        TotalX_trans = ModelX_trans #np.vstack((ModelX_trans,X_m))

        ModelY_trans = (ModelX_trans @ self.mu) # shape(d,)
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
    