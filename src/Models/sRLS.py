import numpy as np
import DataCleaning as DC
from DataObjectModels import ProbPredResultDO
from Models.IModel import IDBdSeqPrbPredModel, IProbaPredResult
from typing import List

__version = "2026.3.5"


class StablisedRLSModel(IDBdSeqPrbPredModel):
    """
    Recursive Least Square Model with stablised version.
    ref Goodwin; Ljung.
    """
    mu: np.ndarray
    P: np.ndarray
    beta: float
    nu: float = 0.5  # nu in logit transform
    lags: list[int] = []  # degree of model
    _is_trained = False

    # # Debug log
    # N = 1000
    # beta_histy :np.array
    # pred_histy = np.empty(N)*np.nan
    # w_histy = np.empty(N)*np.nan
    # muL1 = np.empty(N)*np.nan
    # trace_histy = np.empty(N)*np.nan
    # error_histy = np.empty(N)*np.nan
    trace_trP = []
    
    @property
    def _Parameters(self): return self.mu, self.pR, self.beta

    def __init__(self, lags,mu,P, beta,nu, _lambda,epsilon=0.005,resolution=1001,l1norm=0.1):
        self.lags = lags
        self.mu = mu
        self.P = P
        self.I_P = np.eye(len(mu))
        self.beta = beta
        self.nu = nu
        self._lambda = _lambda
        self.__EPSILON = epsilon
        self.__RESOLUTION = resolution
        self.__l1Norm = l1norm


    def Fit(self, X):

        if (not self._is_trained):
            R = DC.Tranformation.XPrecision(X,lags=self.lags) 
            self.P = np.linalg.inv(R + self.P)

        LogitX = DC.Tranformation.LogitNormal(X, self.nu)
        Y_m = LogitX[:,np.newaxis]
        X_m = DC.Tranformation.SeqToOffsetLagMatrix(LogitX, self.lags) #shape(N,d)
        X_train, Y_train = DC.Tranformation.XYDropNaN(X_m, Y_m) #shape(N,d), (N,1)
        if(len(X_train)==0): return # non valid training X Y pair
        # print(f"X_train:{X_train.shape}")

        for i, (x,y) in enumerate(zip(X_train,Y_train)):
            # print(f"x shape:{x.shape},mu shape:{self.mu.shape}")

            
            ### First calculate the K. then calcualte the update. then update mu and stored P.
            K = self.P @ x / (self._lambda + x.T @ self.P @ x)
            predYmu = x.T @ self.mu
            error = y - predYmu ## scalar
            update = K * error
            change_absmax = np.max(np.abs(update))
            if(change_absmax> self.__l1Norm): 
                # print(f"{i}  y:{y}, pred:{predYmu:.4f} error:{error} x.T:{x.T} mu:{self.mu} update :{update}")
                continue # do not update due to num unstable issue
            
            ### Update mu and P
            self.mu = self.mu + update
            # self.P = 1.0 / lambda [ (self.I_P - K @ x.T) @ self.P ]
            self.P = (self.P - np.outer(K, x) @ self.P) / self._lambda
            # self.trace_P.append(np.trace(self.P))

            ### Update variance
            newPredYmu = (x.T @ self.mu)
            w = 4 * newPredYmu * (1 - newPredYmu)

            if(w<0): w = 0
            if(w>1): w = 1
            _lambda_star = 1 - (1 - self._lambda) * w
            var_error = error ** 2
            self.beta = _lambda_star * self.beta + (1-_lambda_star)*var_error

            if (not self._is_trained): self._is_trained = True
        return

    def ProbaPred(self, X:np.ndarray)-> List[IProbaPredResult]: 

        ### Result Buffer ###
        NumData = len(X)
        result = [None] * NumData

        ### input transformation ###
        LogitX = DC.Tranformation.LogitNormal(X, self.nu)
        X_m = DC.Tranformation.SeqToOffsetLagMatrix(LogitX, self.lags)
        fragment_len = max(self.lags)

        ### each by each prediction ###
        for row in range(NumData):
            Xrow = X_m[row,:]
            predYmu = (Xrow.T@ self.mu)
            # self.pred_histy[row] = predYmu
            predYsigma = np.sqrt(self.beta) 

        ### Output back transformation + Update ###
            if(not np.isnan(predYmu) and not np.isnan(predYsigma)): 
                left = DC.Tranformation.LogitNormal(self.__EPSILON, self.nu)
                right = DC.Tranformation.LogitNormal(1-self.__EPSILON, self.nu)
                trans_pred = ProbPredResultDO.InflatedGaussianPred(mean=predYmu, sigma=predYsigma,left_trunc=left, right_trunc=right)
                origin_quentiles = np.linspace(self.__EPSILON,1-self.__EPSILON,self.__RESOLUTION)
                trans_quentiles = DC.Tranformation.LogitNormal(origin_quentiles,self.nu)
                # print(predYmu,predYsigma,self.nu,left, right, end="\r") # DEUBG
                result[row]= ProbPredResultDO.NumeDoubleBoundProbaPred(origin_quentiles,trans_pred.Cdfs(trans_quentiles))
                self.Fit(X[row-fragment_len:row+1])
        return result



