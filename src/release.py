import numpy as np
import pandas as pd
import os
import scipy
from scipy.stats import norm, ecdf
from scipy.interpolate import PchipInterpolator  as PchipIntp
import statsmodels.tsa.stattools as stt
from collections.abc import Sequence
import numpy as np
import scipy.integrate as integ

Epsilon = 0.005
Resolution = 501

def eCRPS(xs, ys, obs, cdf_obs):
    index_y = np.argmin(xs < obs)
    X_lhs = np.append(xs[0:index_y], obs)
    ys_lhs = np.append(ys[0:index_y], cdf_obs)
    Y_lhs = ys_lhs ** 2
    CRPS_lhs = integ.simpson(Y_lhs, X_lhs)
    X_rhs = np.append(obs, xs[index_y:])
    y_rhs = np.append(cdf_obs, ys[index_y:])
    Y_rhs = (1 - y_rhs) ** 2
    CRPS_rhs = integ.simpson(Y_rhs, X_rhs)
    return CRPS_lhs + CRPS_rhs

def numeCRPS(ppred,obs:float,xmin:float, xmax:float,resolution=1001):
    xs = np.linspace(xmin,xmax,resolution)
    ys = ppred.Cdfs(xs)
    cdf_obs =ppred.Cdf(obs) 
    res = eCRPS(xs,ys,obs,cdf_obs)
    return res

def NanNumeCRPS(ppred,obs:float,xmin:float, xmax:float,resolution=1001):
    if(ppred==None or obs==None or np.isnan(obs)): return np.nan
    return numeCRPS(ppred,obs,xmin,xmax,resolution)

NanNumeCRPSs = np.vectorize(NanNumeCRPS,excluded=['resolution'])

def FindCleanMax(quantiles,spike_threshold_q:float=0.995, max_spike_pcent=10,strictness = 5, strict_continuous=False):
    """
    Try remove spike and find true MAX of windfarm data from its eCDF data.
    Windfarm data has a property that has upper bound. That mean near the true upper bound (Clean Max), the CDF should has a kind of vertical jump.
    Use this property to detect if there is horizontal slay need the end of CDF as spike data.

    quantiles: the quantiles point of empirical CDF (hint, can obtain from scipy.stats.ecdf)
    spike_threshold_q: the quantiles of diff qunatiles set as max accpectable diff quantiles in eCDF data.
    """
    diff_quantiles = quantiles[1:] - quantiles[:-1]
    diff_quantiles_thred = np.quantile(diff_quantiles, spike_threshold_q)
    Count10pcent = np.ceil(len(quantiles) / max_spike_pcent).astype('int')
    CleanMax = quantiles[-1]
    continuous_lower = True
    cleanIndexCounterdown = strictness
    for i in range(1, Count10pcent):
        if diff_quantiles[-i] <= diff_quantiles_thred:
            if not continuous_lower: CleanMax = quantiles[-i]
            continuous_lower = True
            cleanIndexCounterdown -= 1
            if cleanIndexCounterdown == 0: break
        else:
            continuous_lower = False
            if strict_continuous: cleanIndexCounterdown = strictness
            continue
    return CleanMax

def Get(datafilepath, windfarm_name, epsilon, clean_max=None):
    if not os.path.exists(datafilepath):raise AssertionError(f"{datafilepath} not exists")
    rawdata = pd.read_csv(datafilepath)
    if windfarm_name not in rawdata.columns:raise AssertionError("Windfarm should be in data columns")
    data = rawdata[windfarm_name].copy()
    if clean_max is None:clean_max = get_clean_max(data)
    data /= clean_max
    is_scaled = True
    if not is_scaled:raise AssertionError("should rescale data before bounding data")
    data[data < epsilon] = epsilon
    data[data > 1 - epsilon] = 1 - epsilon
    return data.to_numpy().copy(), clean_max

def get_clean_max(data):
    cleandata = data[~np.isnan(data)]
    ecdf_windfarm = ecdf(cleandata)
    quantiles = ecdf_windfarm.cdf.quantiles
    cleanmax = FindCleanMax(quantiles, strictness=5)
    return cleanmax

def GetData(windfarm, train_data_file_path, tset_data_file_path, epsilon=Epsilon):
    X_Train,clean_max = Get(train_data_file_path, windfarm, epsilon)
    X_Test,_ = Get(tset_data_file_path, windfarm, epsilon, clean_max=clean_max)
    return X_Train, X_Test

def LogitNormal(x:np.ndarray,nu:float):
    xnu = x ** nu
    res = np.log(xnu/ (1-xnu))
    return res
    
def InverseLogitNormal(y:np.array, nu:float):
    ey = np.e ** y
    return (ey/(1+ey)) ** (1.0/nu)

def SeqToOffsetLagMatrix(x:np.ndarray|list[float], lags:list[int]):
    """
    Given a 1D sequence x_t , and lags l1..ln, Generate matrix whose i-th row is:
    [1, X_i-l1, X_i-l2, ... , X_i-ln].  
    set np.nan when X_i-lj not exists. 
    """
    length = len(x)
    matrix = np.full((len(x),len(lags)+1),np.nan)
    matrix[:,0] =1.0
    for i,lag in enumerate(lags):
        if lag >= length: continue
        matrix[lag:,i+1] = x[:-lag]
    return matrix

def MatrixDropNanRow(matrix:np.ndarray):
    return matrix[~np.isnan(matrix).any(axis=1), :]

def XYDropNaN(X:np.ndarray,Y:np.ndarray):
    assert X.shape[0] == Y.shape[0], "X Y should have same data count"
    XY = np.concatenate((X,Y),axis=1)
    XY= MatrixDropNanRow(XY)
    X = XY[:,0:X.shape[1]]
    Y = XY[:,X.shape[1]:]
    return X,Y

def XPrecision(X:np.ndarray, lags):
    """
    Given lags such that n_dim = len(lags) + 1. Calculate Precision Matrix shape (n_dim, n_dim).
    N.B.: Some peole call this matrix as precision. 
    """
    X_m = SeqToOffsetLagMatrix(X, lags) #shape (N,d)
    X_m = MatrixDropNanRow(X_m)
    Pcsn = X_m.T @ X_m
    return Pcsn

def SelectOrder(X,max_lags=6):
    filled = np.nan_to_num(X[:5000])
    pacf,confint = stt.pacf(filled,alpha=.05,nlags=max_lags)
    lags = np.where((confint[:,1]<0) | (0<confint[:,0]))[0]
    if(len(lags)<=1): 
        return np.array([1])
    else:
        return lags[1:]

class InflatedGaussianPred():
    '''
    different to truncated, which evenly rescale the pdf, the squeezed one squeeze all proba density 
    outside the bound to THE left/right bound points.
    '''
    mean:float =  0.0
    sigma:float = 1.0
    left_bound:float = -np.inf
    normed_left_bound:float = -np.inf
    cdf_left = 0
    right_bound:float = np.inf
    normed_right_bound:float = np.inf
    cdf_right = 1

    def __init__(self, mean, sigma, left_trunc, right_trunc):
        assert sigma > 0, "sigma should be strict positive"
        self.mean = mean
        self.sigma = sigma
        self.left_bound = left_trunc
        self.normed_left_bound = (left_trunc - mean)/sigma
        self.cdf_left = norm.cdf(self.normed_left_bound)
        self.right_bound = right_trunc
        self.normed_right_bound = (right_trunc- mean)/sigma
        self.cdf_right=norm.cdf(self.normed_right_bound)
    
    def resetMoments(self, mean, sigma):
        self.mean = mean
        self.sigma = sigma
        self.normed_left_bound = (self.left_bound - mean)/sigma
        self.normed_right_bound = (self.right_bound - mean)/sigma
        self.cdf_left = norm.cdf(self.normed_left_bound)
        self.cdf_right=norm.cdf(self.right_bound)     
    
    def Quantile(self, proba:float):
        assert 0 <= proba and proba <= 1, "proba should be in [0,1]"
        quantile =  norm.ppf(proba, loc=self.mean, scale=self.sigma)
        quantile = min(max(quantile,self.left_bound),self.right_bound)
        return quantile

    def Percentile(self, percentage:float):
        assert 0 <= percentage and percentage <= 100, "proba should be in [0,100]"
        proba = percentage / 100.0
        percentile = norm.ppf(proba, loc=self.mean, scale=self.sigma)
        percentile = min(max(percentile,self.left_bound),self.right_bound)
        return percentile

    def Quantiles(self, probas:Sequence[float])->np.array:
        assert 0 <= min(probas) and max(probas) <= 1, "proba should all be in [0,1]"
        res = norm.ppf(probas, loc=self.mean, scale=self.sigma)
        res = np.array([min(max(r,self.left_bound),self.right_bound) for r in res])
        return res
    
    def Percentiles(self, percentages:Sequence[float])->np.array:
        assert 0 <= min(percentages) and max(percentages) <= 100, "proba should all be in [0,100]"
        probas = percentages / 100.0
        res = norm.ppf(probas, loc=self.mean, scale=self.sigma)
        res = np.array([min(max(r,self.left_bound),self.right_bound) for r in res])
        return res
    
    def Cdf(self, quantile:float):
        if(quantile<self.left_bound): return 0
        if(quantile>=self.right_bound): return 1
        res = norm.cdf(quantile, loc=self.mean, scale=self.sigma)
        return res
    
    def Cdfs(self, quantiles:float):
        res = norm.cdf(quantiles, loc=self.mean, scale=self.sigma)
        res[quantiles<self.left_bound] = 0
        res[quantiles>=self.right_bound-1E-10] = 1
        return res

class NumeDoubleBoundProbaPred():
    def __init__(self, quantiles:np.array, cdfs:np.array):
        assert len(quantiles) > 0, "should have non empty data"
        assert len(quantiles)==len(cdfs), "should have same length data"
        self._cdf = PchipIntp(quantiles,cdfs)
        clean_quantiles, clean_cdfs = self.__FilterDuplicated(quantiles,cdfs)
        if(len(clean_cdfs)>=2):
            self._ppf = PchipIntp(clean_cdfs,clean_quantiles)
            self._ppfs = self._ppf
        else: # handle left bound reach case
            self._ppf = lambda x: clean_quantiles[-1]
            self._ppfs = lambda x: np.array([clean_quantiles[0]]*len(x))
        self.left_quantile = min(quantiles)
        self.right_quantile = max(quantiles)

    def Quantile(self, cdf:float)->float:
        assert 0 <= cdf and cdf <= 1, "proba should be in [0,1]"
        return min(max(self._ppf(cdf),0),1)
    
    def Quantiles(self, cdfs:np.array) -> np.array: 
        assert 0 <= min(cdfs) and max(cdfs) <= 1, "proba should all be in [0,1]"
        res = self._ppfs(cdfs)
        res[res>self.right_quantile] = self.right_quantile
        res[res<self.left_quantile] = self.left_quantile
        return res
    
    def Percentile(self, percentage:float)->float:
        assert 0 <= percentage and percentage <= 100, "proba should be in [0,100]"
        return self.Quantile(percentage/100.0)
    
    def Percentiles(self, percentages:np.array)-> np.array: 
        assert 0 <= min(percentages) and max(percentages) <= 100, "proba should all be in [0,100]"
        return self.Quantiles(percentages/100.0)
    
    def Cdf(self, quantile:float)->float: return self._cdf(quantile)       

    def Cdfs(self, quantiles:Sequence[float]):return self._cdf(quantiles)

    def __FilterDuplicated(self,quantiles,cdfs):
        cdfs_diff = np.diff(cdfs)
        index_1 = cdfs_diff > 1E-10
        index = [True]
        index = np.append(index_1,index)
        return quantiles[index], cdfs[index]

class PersistenceModel():
    param: np.ndarray = None
    sigma: float = None
    IsTrained: bool = False

    def __init__(self,epsilon=0.005,resolution=1001):
        self.sigma = 1
        self.__EPSILON = epsilon
        self.__RESOLUTION = resolution

    def Fit(self,X):
        Y_m = X[:,np.newaxis]
        X_m = SeqToOffsetLagMatrix(X, [1])
        X_train, Y_train = XYDropNaN(X_m, Y_m)
        if(len(X_train)==0): return

        data_Y_pred = X_train[:,1]
        residual = data_Y_pred.flatten() - Y_train.flatten()
        self.sigma = np.sqrt(np.average(residual**2))

        self.IsTrained = True

    def Reset(self, lags, nu):
        self.__init__()
        self.IsTrained = False
        self.param = np.zeros((len(lags),1))
        self.sigma = 1

    def ProbaPred(self, X): 
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
    
    def PointPredBounded(self, Z, left, right):
        res = self.PointPred(Z)
        res[res<left] = left
        res[res>right] = right
        return res
    
    def PointPred(self, Z):
        if(not self.IsTrained): Warning("Model not yet trained")
        Z_m = SeqToOffsetLagMatrix(Z, [1])
        res = Z_m[:,1]
        return res
    
class ARModel():
    """
    Notice: param and sigma is in transformed domain. 
            When forecast, forecast a proba dist at transformed doamin,
            Then, transform back the quantile CDF to original doamin.
    """
    lags: list[int] = [1]
    nu: float = 0.5
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
        LogitX = LogitNormal(X, self.nu)
        Y_m = LogitX[:,np.newaxis]
        X_m = SeqToOffsetLagMatrix(LogitX, self.lags)
        X_train, Y_train = XYDropNaN(X_m, Y_m)
        if(len(X_train)==0): return
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

    def ProbaPred(self, X): 
        if(not self.IsTrained): Warning("Model not yet trained")
        LogitZ = LogitNormal(X, self.nu)
        Z_m = SeqToOffsetLagMatrix(LogitZ, self.lags)
        trans_pred = (Z_m @ self.param).flatten()
        nume_res = [None] * len(X)
        trans_left = LogitNormal(self.__EPSILON,self.nu)
        trans_right =LogitNormal(1-self.__EPSILON,self.nu)
        for i in range((len(X))):
            x = trans_pred[i]
            if(np.isnan(x)): continue
            analytic_trans = InflatedGaussianPred(x,self.sigma,trans_left,trans_right) 
            origin_quentiles = np.linspace(self.__EPSILON,1-self.__EPSILON,self.__RESOLUTION)
            trans_quentiles = LogitNormal(origin_quentiles,self.nu)
            nume_res[i] = NumeDoubleBoundProbaPred(origin_quentiles,analytic_trans.Cdfs(trans_quentiles)) 
        return nume_res
    
class ARnuModel():
    """
    AR Fix Logit + nu is obtimised in Fit. No adaptive update during forecasting.
    """
    lags: list[int] = [1]
    nu: float = None
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
        LogitX = LogitNormal(X, self.nu)
        Y_m = LogitX[:,np.newaxis]
        X_m = SeqToOffsetLagMatrix(LogitX, self.lags)
        X_train, Y_train = XYDropNaN(X_m, Y_m)
        if(len(X_train)==0): return

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
        while(downcounter):
            self._partial_fit(X)
            res = self._partial_fit_nu(X,nv_bounds)
            imprv = res.fun - pre_nll
            pre_nll = res.fun
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
    
    def ProbaPred(self, X): 
        if(not self.IsTrained): Warning("Model not yet trained")
        LogitZ = LogitNormal(X, self.nu)
        Z_m = SeqToOffsetLagMatrix(LogitZ, self.lags)
        trans_pred = (Z_m @ self.param).flatten()
        origin_res = [None] * len(X)

        trans_left = LogitNormal(self.__EPSILON,self.nu)
        trans_right =LogitNormal(1-self.__EPSILON,self.nu)
        for i in range((len(X))):
            x = trans_pred[i]
            if(np.isnan(x)): continue
            analytic_trans = InflatedGaussianPred(x,self.sigma,trans_left,trans_right) 
            origin_quentiles = np.linspace(self.__EPSILON,1-self.__EPSILON,self.__RESOLUTION)
            trans_quentiles = LogitNormal(origin_quentiles,self.nu)
            origin_res[i] = NumeDoubleBoundProbaPred(origin_quentiles,analytic_trans.Cdfs(trans_quentiles)) 
        return origin_res

    def NegLogLikelihood(self, X:np.array, nu=None):
        if(not nu): nu = self.nu
        LogitX = LogitNormal(X, nu)
        Y_trans = LogitX[:,np.newaxis]
        Y_origin = X[:,np.newaxis]
        X_m = SeqToOffsetLagMatrix(LogitX, self.lags)
        _, Y = XYDropNaN(X_m, Y_origin)
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
        return negloglh

class RLSModel():
    mu: np.ndarray
    pR: np.ndarray
    beta: float
    nu: float = 0.5
    lags: list[int] = []
    _is_trained = False
    
    @property
    def _Parameters(self): return self.mu, self.pR, self.beta

    def __init__(self, lags,mu,pR, beta,nu=0.5, _lambda= 0.9999,epsilon=0.005,resolution=1001,l1norm=0.1):
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
        if (not self._is_trained):
            self.pR = XPrecision(X,lags=self.lags) 
            self._is_trained = True
            return
        LogitX = LogitNormal(X, self.nu)
        Y_m = LogitX[:,np.newaxis]
        X_m = SeqToOffsetLagMatrix(LogitX, self.lags)
        X_train, Y_train = XYDropNaN(X_m, Y_m)
        if(len(X_train)==0): return
        for i, (x,y) in enumerate(zip(X_train,Y_train)):
            newPrecision = x.T @ x + self._lambda * self.pR
            predYmu = x.T @ self.mu
            error = y - predYmu
            update = np.linalg.solve(newPrecision, x.T)* error[0]
            change_absmax = np.max(np.abs(update))
            if(change_absmax> self.__l1Norm): continue
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
        return

    def ProbaPred(self, X:np.ndarray): 
        NumData = len(X)
        result = [None] * NumData
        LogitX = LogitNormal(X, self.nu)
        X_m = SeqToOffsetLagMatrix(LogitX, self.lags)
        fragment_len = max(self.lags)
        for row in range(NumData):
            Xrow = X_m[row,:]
            predYmu = (Xrow.T@ self.mu)
            predYsigma = np.sqrt(self.beta) 
            if(not np.isnan(predYmu) and not np.isnan(predYsigma)): 
                left = LogitNormal(self.__EPSILON, self.nu)
                right = LogitNormal(1-self.__EPSILON, self.nu)
                trans_pred = InflatedGaussianPred(mean=predYmu, sigma=predYsigma,left_trunc=left, right_trunc=right)
                origin_quentiles = np.linspace(self.__EPSILON,1-self.__EPSILON,self.__RESOLUTION)
                trans_quentiles = LogitNormal(origin_quentiles,self.nu)
                result[row]= NumeDoubleBoundProbaPred(origin_quentiles,trans_pred.Cdfs(trans_quentiles))
                self.Fit(X[row-fragment_len:row+1])
        return result

class NWModel():
    """
    reprod of Adaptive Generalized Logit-Normal Distributions for Wind Power Short-Term Forecasting
    @Amandine Pierrot, 2021
    """
    mu: np.ndarray
    var_z: float
    nu: float = 1
    lags: list[int] = []
    _train_countdown = 200

    def __init__(self, lags,mu,var_z,R:np.ndarray,nu=1, alpha= 0.9999,epsilon=0.005,resolution=1001):
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
            h= self._h(X)[:,np.newaxis]
            R_crm = h @ h.T
            newR = self._alpha * self.R + (1-self._alpha)*R_crm
            if(self._train_countdown>0):
                continue
            change = np.linalg.solve(newR,h.flatten())
            change_l1 = np.sum(np.abs(change))
            if(change_l1>1): continue
            new_param = self.Params + (1-self._alpha)*change 
            new_param = new_param.flatten()
            self.R = newR
            self.mu = new_param[0:self.AR_dim]
            self.var_z = new_param[-2]
            self.nu = new_param[-1]
        return
    
    def ProbaPred(self, X:np.ndarray): 
        NumData = len(X)
        result = [None] * NumData
        fragment_len = max(self.lags)
        for row in range(NumData):
            if(row<= fragment_len): continue
            x_predictor_y = X[row-fragment_len:row+1]
            LogitX = LogitNormal(x_predictor_y, self.nu)
            X_m = SeqToOffsetLagMatrix(LogitX, self.lags)
            Xrow = X_m[-1,:] 
            predYmu = (Xrow.T@ self.mu)
            predYsigma = np.sqrt(self.var_z) 
            if(not np.isnan(predYmu) and not np.isnan(predYsigma)): 
                left = LogitNormal(self.__EPSILON, self.nu)
                right = LogitNormal(1-self.__EPSILON, self.nu)
                trans_pred = InflatedGaussianPred(mean=predYmu, sigma=predYsigma,left_trunc=left, right_trunc=right)
                origin_quentiles = np.linspace(self.__EPSILON,1-self.__EPSILON,self.__RESOLUTION)
                trans_quentiles = LogitNormal(origin_quentiles,self.nu)
                result[row]= NumeDoubleBoundProbaPred(origin_quentiles,trans_pred.Cdfs(trans_quentiles))  
                self.Fit(x_predictor_y)
        return result
    
    def _h(self,X:np.ndarray):
        LogitX = LogitNormal(X, self.nu)
        Y_m = LogitX[:,np.newaxis]
        X_m = SeqToOffsetLagMatrix(LogitX, self.lags)
        X_train, Y_train = XYDropNaN(X_m, Y_m)
        Neffect = len(X_train)
        Y_array = Y_train.flatten()
        delta_theta = (X_train.T @ (Y_array - X_train @ self.mu)) /self.var_z
        pred_error = Y_array - X_train @ self.mu
        delta_varz = - Neffect/self.var_z + pred_error.T @ pred_error / (self.var_z**2)
        delta_nu_1 = - Neffect / self.nu
        ln_x = np.log(X)
        Xnu = X**self.nu
        delta_nu_2 = - np.nansum(ln_x * Xnu / (1-Xnu))
        U = np.log(X) /(1-Xnu)
        U_col = U[:,np.newaxis]
        U_m = SeqToOffsetLagMatrix(U, self.lags)
        U_train, Ucoltrain = XYDropNaN(U_m, U_col)
        U_array = Ucoltrain.flatten()
        U_error = U_array - U_train @ self.mu
        dlta_nu_3 = - U_error.T @ pred_error / self.var_z
        delta_nu = delta_nu_1 + delta_nu_2 + dlta_nu_3
        h = np.hstack((delta_theta,delta_varz,delta_nu))
        return h

class BayesModel():
    mu: np.ndarray
    Precision: np.ndarray
    alpha: float = None
    beta: float = None
    nu: float = 0.5

    lags: list[int] = [1]
    IsTrained: bool = False

    @property
    def PrecisionZ(self): return (self.alpha - 1)/self.beta

    def __init__(self, lags,mu,precision, alpha=101,beta=1,nu=0.5, epsilon=0.005, resolution=1001):
        assert len(mu.shape)==1, "Mu should be 1D vector"
        self.lags = lags
        self.mu = mu
        self.Precision = precision
        self.alpha = alpha
        self.beta = beta
        self.nu = nu
        self.__EPSILON = epsilon
        self.__RESOLUTION = resolution

    def Fit(self, X):
        LogitX = LogitNormal(X, self.nu)
        Y_m = LogitX[:,np.newaxis]
        X_m = SeqToOffsetLagMatrix(LogitX, self.lags)
        X_train, Y_train = XYDropNaN(X_m, Y_m)
        if(len(X_train)==0): return
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
        self.Precision *= 0.999 # 0.9999
        if(np.linalg.det(self.Precision)> 1E20):
            self.Precision *= 0.995
        if(self.alpha<1E20 and self.beta<1E20): pass
        else:
            self.alpha *= 0.9995
            self.beta *= 0.9995
        return

    def ProbaPred(self, X:np.ndarray): 
        NumData = len(X)
        result = [None] * NumData
        LogitX = LogitNormal(X, self.nu)
        X_m = SeqToOffsetLagMatrix(LogitX, self.lags)
        fragment_len = max(self.lags)
        for row in range(NumData):
            Xrow = X_m[row,:]
            predYmu = (Xrow.T@ self.mu).flatten()[0]
            varModel = Xrow.T@ np.linalg.solve(self.Precision,Xrow)
            varNoise = 1/self.PrecisionZ
            if(np.isnan(predYmu) or np.isnan(varModel) or np.isnan(varNoise)):  continue
            predYsigma = np.sqrt( varModel + varNoise) 
            left = LogitNormal(self.__EPSILON, self.nu)
            right = LogitNormal(1-self.__EPSILON, self.nu)
            trans_pred = InflatedGaussianPred(mean=predYmu, sigma=predYsigma,left_trunc=left, right_trunc=right)
            origin_quentiles = np.linspace(self.__EPSILON,1-self.__EPSILON,self.__RESOLUTION)
            trans_quentiles = LogitNormal(origin_quentiles,self.nu)
            result[row]= NumeDoubleBoundProbaPred(origin_quentiles,trans_pred.Cdfs(trans_quentiles))
            self._propagonde()
            self.Fit(X[row-fragment_len:row+1])
        return result

class BayesNuModel():
    mu: np.ndarray
    p:int
    Precision: np.ndarray
    L: np.ndarray
    alpha: float = None
    beta: float = None
    nu: float = 0.5

    lags: list[int] = [1]
    IsTrained: bool = False
    _nus:np.ndarray
    _max_eigval:np.ndarray
    _min_eigval:np.ndarray
    _precision:np.ndarray
    _L_update_succes : bool

    @property
    def PrecisionZ(self): return max((self.alpha - 1)/self.beta, 1E-8)

    def __init__(self, lags,mu,precision, alpha=11,beta=.1,nu=1, epsilon=0.005, resolution=1001,bounds=(0.1,3)):
        self.lags = lags
        assert(len(mu.shape)==1), "mu should be a 1D vector"
        self.mu = mu
        self.p = len(mu)
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
        LogitX = LogitNormal(X, self.nu)
        Y_m = LogitX[:,np.newaxis]
        X_m = SeqToOffsetLagMatrix(LogitX, self.lags)
        X_train, Y_train = XYDropNaN(X_m, Y_m)
        if(len(X_train)==0): return

        newPrecision = X_train.T @ X_train * self.PrecisionZ + self.Precision
        try:
            newL = np.linalg.cholesky(newPrecision)
            self.L = newL
            self._L_update_succes = True
        except np.linalg.LinAlgError:
            self._L_update_succes = False
            return
        newMu = np.linalg.solve(newPrecision, X_train.T @ Y_train * self.PrecisionZ +  self.Precision @ self.mu[:,np.newaxis]).flatten()
        newAlpha = self.alpha + len(X_train) / 2
        newBeta = self.beta + ( Y_train.T @ Y_train + self.mu.T @ self.Precision @ self.mu / self.PrecisionZ  - newMu.T @ newPrecision @ newMu / self.PrecisionZ )[0,0] /2

        self.Precision = newPrecision
        self.mu = newMu
        self.alpha = newAlpha
        self.beta = newBeta
        res = scipy.optimize.minimize(
            lambda nu_: self.NegLogLikelihood(X,nu_[0])+self.VariNegLogLikelihood(nu_[0]),
            x0=self.nu,method='L-BFGS-B', bounds=[self.nu_bound], options=dict(maxiter=20))
        new_nu =  res.x[0]
        lr = 0.05
        self.nu = (1-lr)*self.nu + lr * new_nu

        return
    
    def _propagonde(self):
        if(not self._L_update_succes): 
            self.Precision = self.Precision*0.9 + 0.1*np.eye(len(self.mu))
            return
        self.Precision *= 0.995
        self.alpha *= 0.995
        self.beta *= 0.995
        return

    def ProbaPred(self, X:np.ndarray): 
        NumData = len(X)
        result = [None] * NumData
        self._nus = [self.nu] * NumData
        fragment_len = max(self.lags)
        for row in range(NumData):
            if(row<= fragment_len): continue
            x_predictor_y = X[row-fragment_len:row+1]
            LogitX = LogitNormal(x_predictor_y, self.nu)
            X_m = SeqToOffsetLagMatrix(LogitX, self.lags)
            Xrow = X_m[-1,:] 
            predYmu = (Xrow.T @ self.mu).flatten()[0]
            varModel = Xrow.T @ np.linalg.solve(self.Precision,Xrow)
            varNoise = 1/self.PrecisionZ
            predYsigma = np.sqrt(varModel + varNoise) 
            if(not np.isnan(predYmu) and not np.isnan(predYsigma)): 
                left = LogitNormal(self.__EPSILON, self.nu)
                right = LogitNormal(1-self.__EPSILON, self.nu)
                trans_pred = InflatedGaussianPred(mean=predYmu, sigma=predYsigma,left_trunc=left, right_trunc=right)
                origin_quentiles = np.linspace(self.__EPSILON,1-self.__EPSILON,self.__RESOLUTION)
                trans_quentiles = LogitNormal(origin_quentiles,self.nu)
                result[row]= NumeDoubleBoundProbaPred(origin_quentiles,trans_pred.Cdfs(trans_quentiles))  
                self._propagonde()
                self.Fit(x_predictor_y)
                if(not self._L_update_succes):print(f"Lupdate fail at data index {row}")

            self._nus[row] = self.nu
        return result
    
    def NegLogLikelihood(self, X, nu=None):
        if not nu: nu = self.nu
        LogitX = LogitNormal(X, nu)
        Y_trans = LogitX
        Y_origin = X[:, np.newaxis]
        X_m = SeqToOffsetLagMatrix(LogitX, self.lags)
        _, Y = XYDropNaN(X_m, Y_origin)
        N = Y.shape[0]
        sigma = np.sqrt(1 / self.PrecisionZ)
        termA1 = N * 0.5 * np.log(2 * np.pi)
        termA2 = N * np.log(sigma)
        termB1 = -1 * N * np.log(nu)
        Ynu = Y**nu
        termB2 = np.nansum(np.log(Y * (1 - Ynu)))
        pred = X_m @ self.mu
        resd_term = 0.5 * (pred - Y_trans)**2 / sigma**2
        assert resd_term.shape == Y_trans.shape
        termC = np.nansum(resd_term)
        negloglh = termA1 + termA2 + termB1 + termB2 + termC
        return negloglh
    
    def VariNegLogLikelihood(self, nu=None):
        if not nu: nu = self.nu
        ModelX_trans = self.L.T
        max_X = np.max(np.abs(ModelX_trans))
        normalisor = max_X
        thresd = np.max([LogitNormal(1-2*self.__EPSILON, self.nu), -LogitNormal(2*self.__EPSILON, self.nu)])
        if max_X > thresd:
            normalisor = thresd / max_X
            ModelX_trans *= normalisor
        TotalX_trans = ModelX_trans
        ModelY_trans = ModelX_trans @ self.mu
        ModelY = InverseLogitNormal(ModelY_trans, self.nu)
        ModelY_trans_new = LogitNormal(ModelY, nu)
        TotalY_trans = ModelY_trans_new
        sigma = np.sqrt(1 / self.PrecisionZ)
        pred = TotalX_trans @ self.mu
        residuals = pred - TotalY_trans
        resd_term = 0.5 * residuals**2 / sigma**2
        assert resd_term.shape == TotalY_trans.shape, "wrong prediction shape"
        termC = np.nansum(resd_term)
        negloglh = termC
        return negloglh
    









