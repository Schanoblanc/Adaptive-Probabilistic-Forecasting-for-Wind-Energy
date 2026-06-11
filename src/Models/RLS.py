import numpy as np
import DataCleaning as DC
from DataObjectModels import ProbPredResultDO
from Models.IModel import IDBdSeqPrbPredModel, IProbaPredResult
from typing import List

__version = "2024.4.22"


class RLSModel(IDBdSeqPrbPredModel):
    """
    Recursive Least Square Model
    ref Pinson 2012
    """
    mu: np.ndarray
    pR: np.ndarray
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
    
    @property
    def _Parameters(self): return self.mu, self.pR, self.beta

    def __init__(self, lags,mu,pR, beta,nu, _lambda,epsilon=0.005,resolution=1001,l1norm=0.1):
        self.lags = lags
        self.mu = mu
        self.pR = pR
        self.beta = beta
        self.nu = nu
        self._lambda = _lambda
        self.__EPSILON = epsilon
        self.__RESOLUTION = resolution
        self.__l1Norm = l1norm


    def Fit(self, X):

        ### Very Tricky issue to handle initialisation of Pr, to avoid numerical issue...
        if (not self._is_trained):
            self.pR = DC.Tranformation.XPrecision(X,lags=self.lags) 
            self._is_trained = True
            return

        LogitX = DC.Tranformation.LogitNormal(X, self.nu)
        Y_m = LogitX[:,np.newaxis]
        X_m = DC.Tranformation.SeqToOffsetLagMatrix(LogitX, self.lags) #shape(N,d)
        X_train, Y_train = DC.Tranformation.XYDropNaN(X_m, Y_m) #shape(N,d), (N,1)
        if(len(X_train)==0): return # non valid training X Y pair
        # print(f"X_train:{X_train.shape}")

        for i, (x,y) in enumerate(zip(X_train,Y_train)):
            # print(f"x shape:{x.shape},mu shape:{self.mu.shape}")
            newPrecision = x.T @ x + self._lambda * self.pR

            predYmu = x.T @ self.mu
            error = y - predYmu
            update = np.linalg.solve(newPrecision, x.T)* error[0]
            # print(error,np.round(update,8))
            change_absmax = np.max(np.abs(update))
            if(change_absmax> self.__l1Norm): 
                # print(f"{i}  new pR:{newPrecision.flatten()}, new pR cond:{np.linalg.cond(newPrecision):.4f}, y:{y}, pred:{predYmu:.4f} error:{error} x.T:{x.T} update :{update} updated mu:{newMu} new muL1:{muL1:.4f}")
                continue # do not update due to num unstable issue

            newMu = self.mu + update
            self.pR = newPrecision
            self.mu = newMu

            newPredYmu = (x.T @ self.mu)
            w = 4 * newPredYmu * (1 - newPredYmu)

            if(w<0): w = 0
            if(w>1): w = 1
            _lambda_star = 1 - (1 - self._lambda) * w
            var_error = error ** 2
            self.beta = _lambda_star * self.beta + (1-_lambda_star)*var_error

            # self.w_histy[i] = w
            # self.error_histy[i] = error[0]
            # self.pred_histy[i] = predYmu
            # self.trace_histy[i] = np.trace(self.pR)
            # self.muL1[i] = np.sum(np.abs(self.mu))
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


    def LoggingFit(self, X): # V240802
        LogitX = DC.Tranformation.LogitNormal(X, self.nu)
        Y_m = LogitX[:,np.newaxis]
        X_m = DC.Tranformation.SeqToOffsetLagMatrix(LogitX, self.lags) #shape(N,d)
        # X_train, Y_train = DC.Tranformation.XYDropNaN(X_m, Y_m) #shape(N,d), (N,1)
        # print(f"X_train:{X_train.shape}")
        NumData = len(X)
        mu_0 = [np.nan] * NumData
        mu_1 = [np.nan] * NumData
        beta = [np.nan] * NumData

        for i, (x,y) in enumerate(zip(X_m,Y_m)):
            ### check validity
            if(np.isnan(y) or np.isnan(x).any()): 
                mu_0[i] = np.nan
                mu_1[i] = np.nan
                beta[i] = np.nan
                continue # do not update due to num unstable issue

            # print(f"x shape:{x.shape},mu shape:{self.mu.shape}")
            newPrecision = x.T @ x + self._lambda * self.pR

            predYmu = x.T @ self.mu
            error = y - predYmu
            update = np.linalg.solve(newPrecision, x.T * error[0])
            newMu = self.mu + update
            muL1 = np.sum(np.abs(newMu))
            # print(f"{i}  new pR:{newPrecision.flatten()}, new pR cond:{np.linalg.cond(newPrecision):.4f}, y:{y}, pred:{predYmu:.4f} error:{error} x.T:{x.T} update :{update} updated mu:{newMu} new muL1:{muL1:.4f}")
            if(muL1>50): 
                mu_0[i] = np.nan
                mu_1[i] = np.nan
                beta[i] = np.nan
                continue # do not update due to num unstable issue

            self.pR = newPrecision
            self.mu = newMu

            newPredYmu = (x.T @ self.mu)
            w = 4 * newPredYmu * (1 - newPredYmu)

            if(w<0): w = 0
            if(w>1): w = 1
            _lambda_star = 1 - (1 - self._lambda) * w
            var_error = error ** 2
            self.beta = _lambda_star * self.beta + (1-_lambda_star)*var_error


            # self.w_histy[i] = w
            # self.error_histy[i] = error[0]
            # self.pred_histy[i] = predYmu
            # self.trace_histy[i] = np.trace(self.pR)
            # self.muL1[i] = np.sum(np.abs(self.mu))
            mu_0[i] = self.mu[0]
            mu_1[i] = self.mu[1]
            beta[i] = self.beta[0]
        return mu_0,mu_1,beta

    def LoggingProbaPred(self, X:np.ndarray)-> List[IProbaPredResult]:  # V240802

        ### Result Buffer ###
        NumData = len(X)
        result = [None] * NumData
        mu_0 = [np.nan] * NumData
        mu_1 = [np.nan] * NumData
        beta = [np.nan] * NumData

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
                mu_0[row] = self.mu[0]
                mu_1[row] = self.mu[1]
                beta[row] = self.beta[0]
        return result,mu_0,mu_1,beta
    


