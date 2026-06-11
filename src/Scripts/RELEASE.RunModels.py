import os
import pandas as pd
import numpy as np
import time
import multiprocessing as mp
from multiprocessing.managers import BaseManager

import _bootstrap
import Configuration

###### Path Configuration ######
Root_Folder = Configuration.ROOT_FOLDER
Data_Folder = Configuration.DATA_FOLDER
output_folder = Configuration.OUTPUT_FOLDER 


###### Pmport Personal Packages ######
import Domain
import DataCleaning as DC
import Models.Persistent as Persistent
import Models.ARFixLogit as ARFixLogit
import Models.ARFlxLogit as ARFlxLogit
import Models.BayesFix as BayesFix
import Models.BayesFlx as BayesFlx
import Models.RLS as RLS
import Models.sRLS as sRLS
import Models.RMLE as RMLE
import Models.sRMLE as sRMLE
from DataCleaning import ResultPostProcess as ResPro
from Evaluation import EvaluateCRPS,EvaluatePP,ModelSelection,EvaluationDieboldMariana

###### Basic Model Configuration ######
###### Default Hyper Parameter ######
Epsilon = 0.005
Resolution = 501
MaxOrder = 6
Default_Lambda = 0.9995


###### Evaluation Related Configuration #####
# BoostrapSamplingN = 200 # 500 to avoid memory issue in my FlowX
PPPlot_Resolution = 100


###### Dataset and Models Configuration ######
Persistence_Name = "Persistent"
Ardefault_Name = "AR Logit Default"
Arflx_Name = "AR Flx Logit"
RLS_Name = "RLS"
SRLS_Name = "sRLS"
RMLE_Name = "RMLE"
SRMLE_Name = "sRMLE"
Bayesian_Name = "Bayes"
BayesianAda_Name = "Ada Bayes"
models=[
        Persistence_Name,
        Ardefault_Name,
        Arflx_Name,
        RLS_Name,
        SRLS_Name,
        RMLE_Name,
        SRMLE_Name,
        Bayesian_Name,
        BayesianAda_Name,
]


###### State Class for Parallisation ######
class MyState():
    def __init__(self, data_file_path, data_valid_file_path, windfarmlist, models, PPN=PPPlot_Resolution):
        self.train_aider : DC.DFrameAider.DFrameAider= DC.DFrameAider.DFrameAider().Unverbose().Load(data_file_path)
        self.test_aider : DC.DFrameAider.DFrameAider= DC.DFrameAider.DFrameAider().Unverbose().Load(data_valid_file_path)
        self.metric = self.__CreateMetricTable(windfarmlist, models)
        self.ppplot_res = self.__CreateResultTable("PPPlot",windfarmlist, models,PPN, is_extended=False)
        self.dm_columns = ["DM_n", "DM_mean","DM_std", "DM_statistics"]
        self.dm_metric = self.__CreateDMMeticTable(windfarmlist, models)
        self.order_selection = self.__CreateResultTable("Order_Select_",windfarmlist, models, MaxOrder, is_extended=False)
        self.lock = mp.Lock()

    def __CreateMetricTable(self,windfarmlist, models):
        columns = ["CRPS","Skill_Avg"]
        index = pd.MultiIndex.from_product([windfarmlist,models])
        df = pd.DataFrame(columns=columns, index=index)
        return df
    
    def __CreateDMMeticTable(self,windfarmlist, models):
        columns = self.dm_columns
        index = pd.MultiIndex.from_product([windfarmlist,models,models])
        df = pd.DataFrame(columns=columns, index=index)
        return df
    
    def __CreateResultTable(self, result_name, windfarmlist, models,N,is_extended=True):
        columns = [result_name]
        sampleC_col = [f"{result_name}{i+1}" for i in range(N)]
        if(is_extended):columns.extend(sampleC_col)
        else: columns = sampleC_col
        index = pd.MultiIndex.from_product([windfarmlist,models])
        df = pd.DataFrame(columns=columns, index=index)
        return df

    def TestWindFarmExist(self, windfarm):
        inTrainset = self.train_aider.TestWindfarmExist(windfarm)
        inValidset = self.test_aider.TestWindfarmExist(windfarm)
        return inTrainset and inValidset

    def GetTrainData(self, windfarm): 
        with self.lock:
            X = self.train_aider.SelectWindFarm(windfarm).ScaleByCleanMax().BoundData(Epsilon).ToMatrixAider().GetData()
        return X
    
    def GetTestData(self,windfarm): 
      with self.lock:
        clean_max = self.train_aider.SelectWindFarm(windfarm).GetCleanMax()
        X = self.test_aider.SelectWindFarm(windfarm).ScaleBy(clean_max).BoundData(Epsilon).ToMatrixAider().GetData()
        # print("clean_max",clean_max)
        return X

    def GetMetric(self): return self.metric
    def GetPPPlot(self): return self.ppplot_res
    def GetSharpness(self): return self.sharpness_res
    def GetDMMetric(self): return self.dm_metric
    def GetOrderSelection(self): return self.order_selection
    def SetMetric(self,windfarm,model_name,data):
        with self.lock: self.metric.loc[(windfarm,model_name)] = data
    def SetPPPlot(self,windfarm,model_name,data): 
        with self.lock: self.ppplot_res.loc[(windfarm,model_name)] = data
    def SetSharpness(self,windfarm,model_name,data):
        with self.lock: self.sharpness_res.loc[(windfarm,model_name)] = data
    def SetDMMetric(self,windfarm,datadict):
      columns = self.dm_columns
      with self.lock: 
        mask = self.dm_metric.index.get_level_values(0) == windfarm

        # Build DataFrame from dict
        temp_df = pd.DataFrame.from_dict(datadict, orient="index", columns=columns)

        # temp_df index = (m1, m2), so add windfarm as first level
        temp_df.index = pd.MultiIndex.from_tuples(
            [(windfarm, m1, m2) for (m1, m2) in temp_df.index],
            names=self.dm_metric.index.names
        )

        # Assign
        self.dm_metric.loc[mask, columns] = temp_df
      return
    def SetOrderSelection(self,windfarm,lags):
        data = np.array([np.nan]*MaxOrder)
        index = lags - 1
        data[index] = 1
        with self.lock: self.order_selection.loc[(windfarm)] = data

class MyManager(BaseManager): pass
MyManager.register('MyState', MyState)

def EvaluationWrapper(preds:np.ndarray,refs:np.ndarray,benchmark_CRPSs:np.ndarray):
    CRPSs= EvaluateCRPS.NanNumeCRPSs(preds,refs,Epsilon,1-Epsilon,resolution=Resolution)
    CRPSs_clean = CRPSs[~np.isnan(CRPSs)]
    CRPS_avg = np.average(CRPSs_clean)
    CRPS_avg_bench = np.nanmean(benchmark_CRPSs)
    Skill_avg = (CRPS_avg_bench-CRPS_avg)/(CRPS_avg_bench - 0)
    clean_Y_pred, clean_Y_ref = ResPro.PairFilterOutEmpty(preds,refs) 
    return CRPSs, CRPS_avg, Skill_avg, clean_Y_pred, clean_Y_ref



###############################
###### The Big Loop Func ######
###############################
def RUN_WindFarm(windfarm,TrainOnYear, state:MyState):
          
    if(not state.TestWindFarmExist(windfarm)): return
    X_train = state.GetTrainData(windfarm)
    X_test = state.GetTestData(windfarm)
    lags = ModelSelection.SelectOrder(X_train,max_lags=MaxOrder)
    state.SetOrderSelection(windfarm,lags)
    # return

    ###### Model Fitting and Evaluation ######
    t0 = time.time()

    ###### Persistent ######
    persist_model = Persistent.PersistentModel(epsilon=Epsilon,resolution=Resolution)
    persist_model.Fit(X_train)
    persist_pred = persist_model.ProbaPred(X_test)
    persist_CRPSs= EvaluateCRPS.NanNumeCRPSs(persist_pred,X_test,Epsilon,1-Epsilon,resolution=Resolution)
    persist_CRPSs, persist_CRPS_avg, persist_Skill_avg, persist_clean_Y_pred, persist_clean_Y_ref = EvaluationWrapper(persist_pred,X_test,persist_CRPSs)

    #region Other Model

    # ###### DriftAR Default nu ######
    ARFixLogit_model = ARFixLogit.ARFixLogitModel(lags=lags, nu=1, epsilon=Epsilon,resolution=Resolution)
    ARFixLogit_model.Fit(X_train)
    ARFixLogit_pred = ARFixLogit_model.ProbaPred(X_test)
    ARFixLogit_CRPSs, ARFixLogit_CRPS_avg, ARFixLogit_Skill_avg, ARFixLogit_clean_Y_pred, ARFixLogit_clean_Y_ref = EvaluationWrapper(ARFixLogit_pred,X_test,persist_CRPSs)

    ####### AR Flx nu ######
    ARFlxLogit_model = ARFlxLogit.ARFlxLogitModel(lags=lags, inital_nu=1, epsilon=Epsilon,resolution=Resolution)
    ARFlxLogit_model.Fit(X_train,nv_bounds=[0.1,3])
    ARFlxLogit_pred = ARFlxLogit_model.ProbaPred(X_test)
    ARFlxLogit_CRPSs, ARFlxLogit_CRPS_avg, ARFlxLogit_Skill_avg, ARFlxLogit_clean_Y_pred, ARFlxLogit_clean_Y_ref = EvaluationWrapper(ARFlxLogit_pred,X_test,persist_CRPSs)

    ####### Bayesian with best nu ######
    nu = ARFlxLogit_model.nu
    initmu = ARFlxLogit_model.param.flatten()
    numDim= len(lags) + 1
    precision = np.eye(numDim)/10000
    bayes_m = BayesFix.BayesFixLogitModel(lags=lags,mu=initmu, precision=precision,nu=nu, _forget=Default_Lambda, epsilon=Epsilon,resolution=Resolution)
    bayes_m.Fit(X_train)
    bayes_m_pred = bayes_m.ProbaPred(X_test)
    bayes_CRPSs, bayes_CRPS_avg, bayes_Skill_avg, bayes_clean_Y_pred, bayes_clean_Y_ref = EvaluationWrapper(bayes_m_pred,X_test,persist_CRPSs)

    ####### Bayesian Ada ######
    best_nu = ARFlxLogit_model.nu
    numDim= len(lags) + 1
    initmu = ARFlxLogit_model.param.flatten()
    precision = np.eye(numDim)/10000
    bayesAda_m  = BayesFlx.BayesFlxLogitModel(lags=lags,mu=initmu, precision=precision,nu=best_nu,_forget=Default_Lambda, epsilon=Epsilon,resolution=Resolution)
    bayesAda_m.Fit(X_train)
    bayesAda_m_pred = bayesAda_m.ProbaPred(X_test)
    bayesAda_CRPSs, bayesAda_CRPS_avg, bayesAda_Skill_avg,bayesAda_clean_Y_pred, bayesAda_clean_Y_ref = EvaluationWrapper(bayesAda_m_pred,X_test,persist_CRPSs)

    ####### Pinson2012(RLS) with best nu ######
    nu = ARFlxLogit_model.nu
    initmu = ARFlxLogit_model.param.flatten()
    numDim= len(lags) + 1
    precision = np.eye(numDim)
    beta = ARFlxLogit_model.sigma**2 # 0.1
    rls_m  = RLS.RLSModel(lags,initmu,precision,beta,nu,_lambda=Default_Lambda)
    rls_m.Fit(X_train)
    rls_m_pred = rls_m.ProbaPred(X_test)
    rls_CRPSs, rls_avg, rls_Skill_avg, rls_clean_Y_pred, rls_clean_Y_ref = EvaluationWrapper(rls_m_pred,X_test,persist_CRPSs)

    ####### stable RLS with best nu ######
    initmu = ARFlxLogit_model.param.flatten()
    numDim= len(lags) + 1
    precision = np.eye(numDim)
    beta = ARFlxLogit_model.sigma**2 # 0.1
    nu = ARFlxLogit_model.nu
    sRLS_m = sRLS.StablisedRLSModel(lags=lags,mu=initmu,P=precision,beta=beta,nu=nu,_lambda=Default_Lambda)
    sRLS_m.Fit(X_train)
    sRLS_m_pred = sRLS_m.ProbaPred(X_test)
    sRLS_CRPSs, sRLS_avg, sRLS_Skill_avg, sRLS_clean_Y_pred, sRLS_clean_Y_ref = EvaluationWrapper(sRLS_m_pred,X_test,persist_CRPSs)

    ####### RMLE with best nu ######
    mu = ARFlxLogit_model.param.flatten()
    var_z = ARFlxLogit_model.sigma**2
    nu = ARFlxLogit_model.nu
    R = np.eye(len(lags)+3)
    rmle_m = RMLE.RMLEModel(lags=lags,mu=mu,var_z=var_z,R=R,nu=nu,alpha=Default_Lambda)
    rmle_m.Fit(X_train)
    rmle_m_pred = rmle_m.ProbaPred(X_test)
    rmle_CRPSs, rmle_avg, rmle_Skill_avg, rmle_clean_Y_pred, rmle_clean_Y_ref = EvaluationWrapper(rmle_m_pred,X_test,persist_CRPSs)

    ####### stable RMLE with best nu ######
    mu = ARFlxLogit_model.param.flatten()
    var_z = ARFlxLogit_model.sigma**2
    nu = ARFlxLogit_model.nu
    P = np.eye(len(lags)+3)
    sRMLE_m = sRMLE.StabalisedRMLEModel(lags=lags,mu=mu,var_z=var_z,P=P,nu=nu,alpha=Default_Lambda)
    sRMLE_m.Fit(X_train)
    sRMLE_m_pred = sRMLE_m.ProbaPred(X_test)
    sRMLE_CRPSs, sRMLE_avg, sRMLE_Skill_avg, sRMLE_clean_Y_pred, sRMLE_clean_Y_ref = EvaluationWrapper(sRMLE_m_pred,X_test,persist_CRPSs)
    #endregion

    ###### PPPLot ######
    probas = np.linspace(Epsilon,1-Epsilon,PPPlot_Resolution) # note this epsilon is for 0.5% ~ 99.5% CDF!
    persistent_pp = EvaluatePP.PPEvaluation_ProbaPred(persist_clean_Y_ref,persist_clean_Y_pred,probas)
    arfix_pp = EvaluatePP.PPEvaluation_ProbaPred(ARFixLogit_clean_Y_ref,ARFixLogit_clean_Y_pred,probas)
    arflx_pp = EvaluatePP.PPEvaluation_ProbaPred(ARFlxLogit_clean_Y_ref,ARFlxLogit_clean_Y_pred,probas)
    bayes_pp = EvaluatePP.PPEvaluation_ProbaPred(bayes_clean_Y_ref,bayes_clean_Y_pred,probas)
    bayesAda_pp = EvaluatePP.PPEvaluation_ProbaPred(bayesAda_clean_Y_ref,bayesAda_clean_Y_pred,probas)
    rls_pp = EvaluatePP.PPEvaluation_ProbaPred(rls_clean_Y_ref,rls_clean_Y_pred,probas)
    s_rls_pp = EvaluatePP.PPEvaluation_ProbaPred(sRLS_clean_Y_ref,sRLS_clean_Y_pred,probas)
    rmle_pp = EvaluatePP.PPEvaluation_ProbaPred(rmle_clean_Y_ref,rmle_clean_Y_pred,probas)
    srmle_pp = EvaluatePP.PPEvaluation_ProbaPred(sRMLE_clean_Y_ref,sRMLE_clean_Y_pred,probas)

    ###### Result Collection ######
    state.SetMetric(windfarm,Persistence_Name,[persist_CRPS_avg, persist_Skill_avg])
    state.SetMetric(windfarm,Ardefault_Name,[ARFixLogit_CRPS_avg, ARFixLogit_Skill_avg])
    state.SetMetric(windfarm,Arflx_Name,[ARFlxLogit_CRPS_avg, ARFlxLogit_Skill_avg])
    state.SetMetric(windfarm,Bayesian_Name,[bayes_CRPS_avg, bayes_Skill_avg])
    state.SetMetric(windfarm,BayesianAda_Name,[bayesAda_CRPS_avg, bayesAda_Skill_avg])
    state.SetMetric(windfarm,RLS_Name,[rls_avg, rls_Skill_avg])
    state.SetMetric(windfarm,SRLS_Name,[sRLS_avg, sRLS_Skill_avg])
    state.SetMetric(windfarm,RMLE_Name,[rmle_avg, rmle_Skill_avg])
    state.SetMetric(windfarm,SRMLE_Name,[sRMLE_avg, sRMLE_Skill_avg])

    ## Save PPLot result ###
    state.SetPPPlot(windfarm,Persistence_Name,persistent_pp)
    state.SetPPPlot(windfarm,Ardefault_Name,arfix_pp)
    state.SetPPPlot(windfarm,Arflx_Name,arflx_pp)
    state.SetPPPlot(windfarm,RLS_Name,rls_pp)
    state.SetPPPlot(windfarm,SRLS_Name,s_rls_pp)
    state.SetPPPlot(windfarm,RMLE_Name,rmle_pp)
    state.SetPPPlot(windfarm,SRMLE_Name,srmle_pp)
    state.SetPPPlot(windfarm,Bayesian_Name,bayes_pp)
    state.SetPPPlot(windfarm,BayesianAda_Name,bayesAda_pp)

    ###### Diebold Mariana Test result ######
    CRPSdict = {
        Persistence_Name: persist_CRPSs,
        Ardefault_Name: ARFixLogit_CRPSs,
        Arflx_Name: ARFlxLogit_CRPSs,
        RLS_Name: rls_CRPSs,
        SRLS_Name: sRLS_CRPSs,
        RMLE_Name: rmle_CRPSs,
        SRMLE_Name: sRMLE_CRPSs,
        Bayesian_Name: bayes_CRPSs,
        BayesianAda_Name: bayesAda_CRPSs,
    }
    DMresult = EvaluationDieboldMariana.Diebold_Mariana_Statistics_Matrix(CRPSdict,horizon=1)
    state.SetDMMetric(windfarm, DMresult)

    ### Logging
    duration = time.time() - t0
    log1 = f"{windfarm} train {TrainOnYear} test {TrainOnYear+1}, {duration:.2f}sec"

    log2 = f"CRPS        :{Persistence_Name}:{persist_CRPS_avg:.5f},\
    {Ardefault_Name}:{ARFixLogit_CRPS_avg:.5f},\
    {Arflx_Name}:{ARFlxLogit_CRPS_avg:.5f},\
    {RLS_Name}:{rls_avg:.5f},\
    {SRLS_Name}:{sRLS_avg:.5f},\
    {RMLE_Name}:{rmle_avg:.5f},\
    {SRMLE_Name}:{sRMLE_avg:.5f},\
    {Bayesian_Name}:{bayes_CRPS_avg:.5f},\
    {BayesianAda_Name}:{bayesAda_CRPS_avg:.5f}.\
    "

    log3 = f"Skill on Avg:{Persistence_Name}:{persist_Skill_avg:.5f},\
    {Ardefault_Name}:{ARFixLogit_Skill_avg:.5f},\
    {Arflx_Name}:{ARFlxLogit_Skill_avg:.5f},\
    {RLS_Name}:{rls_Skill_avg:.5f},\
    {SRLS_Name}:{sRLS_Skill_avg:.5f},\
    {RMLE_Name}:{rmle_Skill_avg:.5f},\
    {SRMLE_Name}:{sRMLE_Skill_avg:.5f},\
    {Bayesian_Name}:{bayes_Skill_avg:.5f},\
    {BayesianAda_Name}:{bayesAda_Skill_avg:.5f}.\
    "
    print(f"{log1}\n{log2}\n{log3}")
    return
# end of script



if __name__=="__main__":

    os.makedirs(output_folder, exist_ok=True)
    
    ###### Dataset ######
    windfarmlist = Domain.Constant.WINDFARMS
    # windfarmlist = ["2__PSTAT001"] ## For Debug
  
    ###### Train by Year ######
    Train_Years = [2020,2021,2022]

    ###### Train by Year ######
    for TrainYear in Train_Years: 
        data_file_path = os.path.join(Data_Folder, f"UBOR_{TrainYear}.csv")
        data_valid_file_path = os.path.join(Data_Folder, f"UBOR_{TrainYear+1}.csv") # use next year as validation set
  
        tickA = time.time()
        with MyManager() as manager:
            state = manager.MyState(data_file_path,data_valid_file_path,windfarmlist,models)
    
            args = [(windfarm, TrainYear, state) for windfarm in windfarmlist]
            with mp.Pool(4) as pool: pool.starmap(RUN_WindFarm,args)
    
            state.GetOrderSelection().to_csv(os.path.join(output_folder,f"_Order_TrainOn_{TrainYear}.csv"))
            state.GetMetric().to_csv(os.path.join(output_folder,f"_Metric_TestOn_{TrainYear+1}.csv"))
            state.GetPPPlot().to_csv(os.path.join(output_folder,f"_PPPlot_TestOn_{TrainYear+1}.csv"))
            state.GetDMMetric().to_csv(os.path.join(output_folder,f"_DM_Metric_TestOn_{TrainYear+1}.csv"))
        tickB = time.time()
        print(f"Total run in {tickB - tickA :.2f}sec")
  
### Running script (Windows)
__readme__ ="""
cd <Path to the directory of /src>
python "./Scripts/RELEASE.RunModels.py"
"""


