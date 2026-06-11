import numpy as np
from scipy.stats import truncnorm, norm
from scipy.interpolate import PchipInterpolator  as PchipIntp
from collections.abc import Sequence
from abc import ABC, abstractmethod

class IProbaPredResult(ABC):
    @abstractmethod
    def Quantile(self, proba:float)->float: ... # Give a cdf value in (0,1) return the correspondent quantile value
    @abstractmethod
    def Quantiles(self, probas:np.array) -> np.array: ... # list version pf Quantile
    @abstractmethod
    def Percentile(self, percentage:float)->float: ... # Quantile in percentage scale ([0,100])
    @abstractmethod
    def Percentiles(self, percentages:np.array)-> np.array: ... # list version pf Quantile
    @abstractmethod
    def Cdf(self, quantile:float)->float: ... # calculate the cdf of given quantile
    @abstractmethod
    def Cdfs(self, quantiles:np.array)-> np.array: ... # list version of cdf

class TruncatedGaussianPred(IProbaPredResult):
    '''
    https://en.wikipedia.org/wiki/Truncated_normal_distribution 
    '''
    mean:float =  0.0
    sigma:float = 1.0
    left_bound:float = -np.inf
    normed_left_bound:float = -np.inf
    right_bound:float = np.inf
    normed_right_bound:float = np.inf

    def __init__(self, mean, sigma, left_trunc, right_trunc):
        assert sigma > 0, "sigma should be strict positive"
        self.mean = mean
        self.sigma = sigma
        self.left_bound = left_trunc
        self.normed_left_bound = (left_trunc - mean)/sigma
        self.right_bound = right_trunc
        self.normed_right_bound = (right_trunc - mean)/sigma
    
    def resetMoments(self, mean, sigma):
        self.mean = mean
        self.sigma = sigma
        self.normed_left_bound = (self.left_bound - mean)/sigma
        self.normed_right_bound = (self.right_bound - mean)/sigma     
    
    def Quantile(self, proba:float):
        assert 0 <= proba and proba <= 1, "proba should be in [0,1]"
        return truncnorm.ppf(proba, a=self.normed_left_bound, b=self.normed_right_bound, loc=self.mean, scale=self.sigma)
    
    def Percentile(self, percentage:float):
        assert 0 <= percentage and percentage <= 100, "proba should be in [0,100]"
        proba = percentage / 100.0
        return truncnorm.ppf(proba, a=self.normed_left_bound, b=self.normed_right_bound, loc=self.mean, scale=self.sigma)
    
    def Quantiles(self, probas:Sequence[float]):
        assert 0 <= min(probas) and max(probas) <= 1, "proba should all be in [0,1]"
        res = truncnorm.ppf(probas, a=self.normed_left_bound, b=self.normed_right_bound, loc=self.mean, scale=self.sigma)
        return res
    
    def Percentiles(self, percentages:Sequence[float]):
        assert 0 <= min(percentages) and max(percentages) <= 100, "proba should all be in [0,100]"
        probas = percentages / 100.0
        res = truncnorm.ppf(probas, a=self.normed_left_bound, b=self.normed_right_bound, loc=self.mean, scale=self.sigma)
        return res
    
    def Cdf(self, quantile:float):
        res = truncnorm.cdf(quantile,a=self.normed_left_bound, b=self.normed_right_bound, loc=self.mean, scale=self.sigma)
        return res
    
    def Cdfs(self, quantiles:float):
        res = truncnorm.cdf(quantiles,a=self.normed_left_bound, b=self.normed_right_bound, loc=self.mean, scale=self.sigma)
        return res

class InflatedGaussianPred(IProbaPredResult):
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

class NumeDoubleBoundProbaPred(IProbaPredResult):
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

class EmpiricalProbaPred(IProbaPredResult):
    """
    use ecdf as cdf estimation
    """
    def __init__(self, emp_cdf_eval, quantiles:np.array, cdfs:np.array):
        self._cdf = emp_cdf_eval
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

class ProbaMassPred(IProbaPredResult):
    def __init__(self, quantiles:np.array, cdfs:np.array):
        assert len(quantiles) > 0, "should have non empty data"
        assert len(quantiles)==len(cdfs), "should have same length data"
        self._cdf = PchipIntp(quantiles,cdfs)
        clean_cdfs, clean_quantiles = self.__FilterDuplicated(quantiles,cdfs)

        self._ppf = PchipIntp(clean_cdfs,clean_quantiles)
        self.left_quantile = min(quantiles)
        self.right_quantile = max(quantiles)

    def Quantile(self, proba:float)->float:
        assert 0 <= proba and proba <= 1, "proba should be in [0,1]"
        return self._ppf(proba)
    
    def Quantiles(self, probas:np.array) -> np.array: 
        assert 0 <= min(probas) and max(probas) <= 1, "proba should all be in [0,1]"
        return self._ppf(probas)
    
    def Percentile(self, percentage:float)->float:
        assert 0 <= percentage and percentage <= 100, "proba should be in [0,100]"
        return self._ppf(percentage)[0] * 100
    
    def Percentiles(self, percentages:np.array)-> np.array: 
        assert 0 <= min(percentages) and max(percentages) <= 100, "proba should all be in [0,100]"
        return self._ppf(percentages)[0] * 100
    
    def Cdf(self, quantile:float)->float: return self._cdf(quantile)       

    def Cdfs(self, quantiles:Sequence[float]):return self._cdf(quantiles)

    def __FilterDuplicated(self,quantiles,cdfs):
        cdfs_diff = np.diff(cdfs)
        index_1 = cdfs_diff > 1E-10
        index = [index_1[0]]
        index.extend(index_1)
        return cdfs[index],quantiles[index]    

    