import numpy as np
import pandas as pd
import scipy.stats as stats
from DataObjectModels.DataQualityReportDO import DQSum


def DataQualitySummarise(windfarm_data:pd.Series,windfarm:str,year:int,
                         spike_threshold_config:float=0.995,
                         max_spike_pcent=10,epsilon = 0.005,
                         strictness = 5,
                         strict_continuous=False):
    """
    Summarise Scalar Data Quality indicators

    Parameters:
    windfarm_df: the series of windfarm data.
    windfarm: the index name of windfarm
    year:
    spike_threshold_config: the quantiles of diff qunatiles set as max accpectable diff quantiles in eCDF data. Used for find clean max.
    max_spike_pcent: the believed max percentage the spike data could occupy.
    epsilon: the threshold to bound data between [eplison, 1-epsilon]

    Returns:

    """
    cleandata = windfarm_data.dropna()
    ecdf_windfarm = stats.ecdf(cleandata)
    quantiles = ecdf_windfarm.cdf.quantiles
    probas = ecdf_windfarm.cdf.probabilities
    cleanmax = FindCleanMax(quantiles,strictness = strictness,strict_continuous=strict_continuous)
    N = len(windfarm_data)
    zero_count = np.sum(windfarm_data<=0)
    nan_count = np.sum(windfarm_data.isna())
    spikecount = np.sum(cleanmax<windfarm_data)
    epsilon_count = np.sum(windfarm_data<=epsilon*cleanmax)
    clean_epsilon_max_count = np.sum(windfarm_data>= ((1-epsilon)*cleanmax))

    DQsum = DQSum(windfarm,year,spike_threshold_config,epsilon,max_spike_pcent,
                  nan_count,np.round(nan_count / N * 100,2), zero_count,np.round(probas[0]*100,2),spikecount,np.round(spikecount / N  * 100.0,2),
                  epsilon_count,np.round(epsilon_count / N  * 100.0,2), cleanmax, clean_epsilon_max_count,np.round(clean_epsilon_max_count / N  * 100.0,2)
    )
    return DQsum,  DQsum.ToDictionary()



def FindCleanMax(quantiles:pd.DataFrame,spike_threshold_q:float=0.995, max_spike_pcent=10,strictness = 5, strict_continuous=False):
    """
    Try remove spike and find true MAX of windfarm data from its eCDF data.
    Windfarm data has a property that has upper bound. That mean near the true upper bound (Clean Max), the CDF should has a kind of vertical jump.
    Use this property to detect if there is horizontal slay need the end of CDF as spike data.

    quantiles: the quantiles point of empirical CDF (hint, can obtain from scipy.stats.ecdf)
    spike_threshold_q: the quantiles of diff qunatiles set as max accpectable diff quantiles in eCDF data.
    """
    

    diff_quantiles = (quantiles[1:]-quantiles[:-1])
    diff_quantiles_thred= np.quantile(diff_quantiles,spike_threshold_q)
    Count10pcent = np.ceil(len(quantiles)/max_spike_pcent).astype('int')
    CleanMax = quantiles[-1]
    
    continuous_lower = True
    cleanIndexCounterdown = strictness 
    for i in range(1,Count10pcent): # only think max max_spike_pcent% data could be spike.
        if(diff_quantiles[-i] <= diff_quantiles_thred):
            if(not continuous_lower): CleanMax = quantiles[-i] # only update when cross under the thred
            continuous_lower = True
            cleanIndexCounterdown -= 1
            if(cleanIndexCounterdown==0): break # Find the Clean Max
        else: # (diff_quantiles[-i] > diff_quantiles_thred), consider it is still a spike.
            continuous_lower = False
            if(strict_continuous): cleanIndexCounterdown = strictness
            continue
    return CleanMax

