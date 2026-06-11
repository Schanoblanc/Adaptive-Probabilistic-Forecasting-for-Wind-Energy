import numpy as np
import DataCleaning as DC
from DataObjectModels import ProbPredResultDO
from Models.IModel import IDBdSeqPrbPredModel, IProbaPredResult
from typing import List

__version = "2024.4.22"


class RMLEModel(IDBdSeqPrbPredModel):
    """
    This is one implementation for the Recursive Maximum Likelihood Estimation Model
    reprod of Adaptive Generalized Logit-Normal Distributions for Wind Power Short-Term Forecasting
    @Amandine Pierrot, 2021
    * Instead of use default nu=1 as initial value, I used the pre-estimated one to improve stability.
    """
    mu: np.ndarray # shape (d,)
    var_z: float # square of sigma, i.e., varaince
    nu: float = 1  # nu in logit transform
    lags: list[int] = []  # degree of model
    _train_countdown = 200

    def __init__(self, lags,mu,var_z,R:np.ndarray,nu, alpha,epsilon=0.005,resolution=1001):
        self.lags = lags
        self.mu = mu
        self.var_z = var_z
        self.R = R
        self.nu = nu
        self._alpha = alpha
        self.__EPSILON = epsilon
        self.__RESOLUTION = resolution
        self.AR_dim = len(mu)

    @property
    def Params(self):return np.hstack((self.mu,self.var_z,self.nu))

    def Fit(self, X):
        self._train_countdown +=1

        while( self._train_countdown>0):
            self._train_countdown-=1
            # print(self._train_countdown)

            h= self._h(X)[:,np.newaxis] #shape(D,1)
            R_crm = h @ h.T # incremental of R
            newR = self._alpha * self.R + (1-self._alpha)*R_crm
            if(self._train_countdown>0):
                ## the pre-train process has not yet finish
                continue

            # print(np.linalg.det(self.R), np.trace(self.R))
            change = np.linalg.solve(newR,h.flatten())
            change_l1 = np.sum(np.abs(change))
            if(change_l1>1): continue # do not update due to num unstable issue

            new_param = self.Params + (1-self._alpha)*change 
            new_param = new_param.flatten()
            # print(new_param)
            self.R = newR
            self.mu = new_param[0:self.AR_dim]
            self.var_z = new_param[-2]
            self.nu = new_param[-1]

        return
    
    def ProbaPred(self, X:np.ndarray)-> List[IProbaPredResult]: 

        ### Result Buffer ###
        NumData = len(X)
        result = [None] * NumData

        ### each by each prediction ###
        fragment_len = max(self.lags)
        for row in range(NumData):

            ### input transformation ###
            if(row<= fragment_len): continue ## non computation for the starting
            x_predictor_y = X[row-fragment_len:row+1]
            LogitX = DC.Tranformation.LogitNormal(x_predictor_y, self.nu)
            X_m = DC.Tranformation.SeqToOffsetLagMatrix(LogitX, self.lags) #shape(N,d)
            Xrow = X_m[-1,:] 
            predYmu = (Xrow.T@ self.mu)
            predYsigma = np.sqrt(self.var_z) 

        ### Output back transformation + Update ###
            if(not np.isnan(predYmu) and not np.isnan(predYsigma)): 
                left = DC.Tranformation.LogitNormal(self.__EPSILON, self.nu)
                right = DC.Tranformation.LogitNormal(1-self.__EPSILON, self.nu)
                trans_pred = ProbPredResultDO.InflatedGaussianPred(mean=predYmu, sigma=predYsigma,left_trunc=left, right_trunc=right)
                origin_quentiles = np.linspace(self.__EPSILON,1-self.__EPSILON,self.__RESOLUTION)
                trans_quentiles = DC.Tranformation.LogitNormal(origin_quentiles,self.nu)
                # print(predYmu,predYsigma,self.nu,left, right, end="\r") # DEUBG
                result[row]= ProbPredResultDO.NumeDoubleBoundProbaPred(origin_quentiles,trans_pred.Cdfs(trans_quentiles))  
                self.Fit(x_predictor_y)
        return result
    
    def _h(self,X:np.ndarray):
        LogitX = DC.Tranformation.LogitNormal(X, self.nu)
        Y_m = LogitX[:,np.newaxis]
        X_m = DC.Tranformation.SeqToOffsetLagMatrix(LogitX, self.lags) #shape(N,d)
        X_train, Y_train = DC.Tranformation.XYDropNaN(X_m, Y_m) #shape(N,d), (N,1)
        Neffect = len(X_train) # effective valid number of X Y pair
        Y_array = Y_train.flatten()

        delta_theta = (X_train.T @ (Y_array - X_train @ self.mu)) /self.var_z  #shape (d,1)

        pred_error = Y_array - X_train @ self.mu #shape (N,1)
        delta_varz = - Neffect/self.var_z + pred_error.T @ pred_error / (self.var_z**2) # shape (), sigle value

        delta_nu_1 = - Neffect / self.nu
        ln_x = np.log(X)
        Xnu = X**self.nu
        delta_nu_2 = - np.nansum(ln_x * Xnu / (1-Xnu)) # shape (), single value 
        
        U = np.log(X) /(1-Xnu)
        U_col = U[:,np.newaxis]
        U_m = DC.Tranformation.SeqToOffsetLagMatrix(U, self.lags) #shape(N,d)
        U_train, Ucoltrain = DC.Tranformation.XYDropNaN(U_m, U_col)
        U_array = Ucoltrain.flatten()
        U_error = U_array - U_train @ self.mu # shape (N,)
        dlta_nu_3 = - U_error.T @ pred_error / self.var_z
        delta_nu = delta_nu_1 + delta_nu_2 + dlta_nu_3 #shape(), single value

        h = np.hstack((delta_theta,delta_varz,delta_nu))
        return h



