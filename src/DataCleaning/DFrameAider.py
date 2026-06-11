import pandas as pd
import os
from DataCleaning.MarixAider import MatrixAider
from DataCleaning.DataQualities import FindCleanMax
from scipy.stats import ecdf

class DFrameAider:
    _rawdata: pd.DataFrame = None
    _data:pd.Series = None
    _is_timeindex_extended: bool = False
    _is_scaled:bool = False
    _verbose:bool = False

    def __init__(self): pass
    
    def Verbose(self):
        self._verbose = True
        print("turn On verbose")
        return self
    
    def Unverbose(self):
        if(self._verbose): print("turn Off verbose")
        self._verbos = False
        return self

    def Load(self,datafilepath=""):
        if(not os.path.exists(datafilepath)): raise AssertionError(f"{datafilepath} not exists")
        self._rawdata = pd.read_csv(datafilepath)
        return self
    
    def SelectWindFarmWithTime(self):
        ### TODO also select data with time
        raise NotImplementedError()

    def SelectWindFarm(self,Windfarm_Name):
        assert Windfarm_Name in self._rawdata.columns, "Windfarm should in data columns"
        self._data = self._rawdata[Windfarm_Name].copy()
        return self

    def TestWindfarmExist(self,Windfarm_Name): return Windfarm_Name in self._rawdata.columns

    def FillNaByZero(self):
        assert not self._is_timeindex_extended, "FillNanByZero can only make sense before extended time index"
        self._data.fillna(0,inplace=True)

        if(self._verbose):
            count = self._data.where(lambda x : x == 0).count()
            per = count/ len(self._data)
            print(f"By Filling, 0 occupies {per*100 :.2f}% of {len(self._data)} datarow")
        return self
    
    def FillNaByEpsilon(self,epsilon):
        assert not self._is_timeindex_extended, "FillNanByZero can only make sense before extended time index"
        self._data.fillna(epsilon,inplace=True)

        if(self._verbose):
            count = self._data.where(lambda x : x <= epsilon).count()
            per = count/ len(self._data)
            print(f"By Filling, epsilon occupies {per*100 :.2f}% of {len(self._data)} datarow")
        return self
    
    def ScaleByMax(self):
        raise Warning("deprecated Method")
        capacity = self._data.dropna().max()
        self._data /= capacity
        self._is_scaled = True
        return self

    def ScaleByCleanMax(self):
        cleanmax = self.GetCleanMax()
        self.ScaleBy(cleanmax)
        return self
    
    def ScaleBy(self,max):
        cleanmax = max
        self._data /= cleanmax
        self._is_scaled = True
        return self    
    
    def GetCleanMax(self):
        cleandata = self._data.dropna()
        ecdf_windfarm = ecdf(cleandata)
        quantiles = ecdf_windfarm.cdf.quantiles
        cleanmax = FindCleanMax(quantiles,strictness = 5)
        return cleanmax
    
    def BoundData(self,epsilon):
        assert self._is_scaled, "should rescale data before bounding data"
        self._data[self._data<epsilon] = epsilon
        self._data[self._data>1-epsilon] = 1-epsilon
        return self
    
    def GetData(self): return self._data.copy()
    
    def ToMatrixAider(self):
        matrixaider = MatrixAider()
        data = self._data.to_numpy()
        matrixaider.Load(data)
        return matrixaider