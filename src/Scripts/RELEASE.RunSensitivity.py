import os
import sys
import pandas as pd
import numpy as np
import time
import copy
from multiprocessing import Pool

import _bootstrap
import Configuration

###### Path Configuration ######
Root_Folder = Configuration.ROOT_FOLDER
Data_Folder = Configuration.DATA_FOLDER
output_folder = Configuration.OUTPUT_FOLDER 

###### Pmport Personal Packages ######
import DataCleaning as DC

import Models.ARFlxLogit as ARFlxLogit
import Models.RLS as RLS
import Models.sRLS as sRLS
import Models.RMLE as RMLE
import Models.sRMLE as sRMLE
import Models.BayesFix as BayesFix
import Models.BayesFlx as BayesFlx
from Evaluation import EvaluateCRPS,ModelSelection


###### Basic Model Configuration ######
###### Default Hyper Parameter ######
Epsilon = 0.005
Resolution = 501
MaxOrder = 6
Default_Lambda = 0.9995
N_RUN = 120 ### 120

###### Evaluation Related Configuration #####
# BoostrapSamplingN = 200 # 500 to avoid memory issue in my FlowX
PPPlot_Resolution = 100

###### Model HyperParameter
Epsilon = 0.005
Resolution = 501


###### Data Configuration and Loding######
Data_Year = 2020
windfarm ="T_AKGLW-2"
Persistence_CRPS = 0.0363085134118612
# Farm Method CRPS Sill
# T_AKGLW-2	Persistent	0.036308513	0
# T_AKGLW-2	AR Logit Default	0.034712992	0.043943456
# T_AKGLW-2	AR Flx Logit	0.034714088	0.043913253
# T_AKGLW-2	RLS	0.035239786	0.029434617
# T_AKGLW-2	sRLS	0.034316145	0.054873321
# T_AKGLW-2	RMLE	0.034714013	0.043915329
# T_AKGLW-2	sRMLE	0.034713836	0.043920194
# T_AKGLW-2	Bayes	0.034533501	0.048886956
# T_AKGLW-2	Ada Bayes	0.034130741	0.059979651





Data_Year_Valid = Data_Year + 1
data_file_path = os.path.join(Data_Folder, f"UBOR_{Data_Year}.csv")
data_valid_file_path = os.path.join(Data_Folder, f"UBOR_{Data_Year_Valid}.csv")
Train_Data_Aider = DC.DFrameAider.DFrameAider().Load(data_file_path)
Test_Data_Aider = DC.DFrameAider.DFrameAider().Load(data_valid_file_path)
def LoadTrainTest(windfarm) : 
    X_train = Train_Data_Aider.SelectWindFarm(windfarm).ScaleByCleanMax().BoundData(Epsilon).ToMatrixAider().GetData()
    clean_max = Train_Data_Aider.SelectWindFarm(windfarm).GetCleanMax()
    X_test = Test_Data_Aider.SelectWindFarm(windfarm).ScaleBy(clean_max).BoundData(Epsilon).ToMatrixAider().GetData()
    return X_train,X_test
X_train,X_test= LoadTrainTest(windfarm)

### arflx benchmark
lags = ModelSelection.SelectOrder(X_train,max_lags=6)
arflx = ARFlxLogit.ARFlxLogitModel(lags=lags)
arflx.Fit(X_train)

###### Train Wrapper ######
def TrainRLS():
    nu = arflx.nu
    initmu = arflx.param.flatten()
    numDim= len(lags) + 1
    precision = np.eye(numDim)
    beta = arflx.sigma**2
    rls_m  = RLS.RLSModel(lags,initmu,precision,beta,nu,_lambda=Default_Lambda)
    rls_m.Fit(X_train)
    return rls_m

def TrainSRLS():
    nu = arflx.nu
    initmu = arflx.param.flatten()
    numDim= len(lags) + 1
    precision = np.eye(numDim)
    beta = arflx.sigma**2
    sRLS_m = sRLS.StablisedRLSModel(lags=lags, mu=initmu, P=precision, beta=beta, nu=nu, _lambda=Default_Lambda)
    sRLS_m.Fit(X_train)
    return sRLS_m

def TrainRMLE():
    nu = arflx.nu
    mu = arflx.param.flatten()
    var_z = arflx.sigma**2
    R = np.eye(len(lags)+3)
    rmle = RMLE.RMLEModel(lags=lags, mu=mu, var_z=var_z, R=R, nu=nu, alpha=Default_Lambda)
    rmle.Fit(X_train)
    return rmle

def TrainSRMLE():
    mu = arflx.param.flatten()
    var_z = arflx.sigma**2
    nu = arflx.nu
    P = np.eye(len(lags)+3)
    sRMLE_m = sRMLE.StabalisedRMLEModel(lags=lags, mu=mu, var_z=var_z, P=P, nu=nu, alpha=Default_Lambda)
    sRMLE_m.Fit(X_train)
    return sRMLE_m

def TrainBayes():
    nu = arflx.nu
    initmu = arflx.param.flatten()
    numDim= len(lags) + 1
    precision = np.eye(numDim)/10000
    bayes_m = BayesFix.BayesFixLogitModel(lags=lags,mu=initmu, precision=precision,nu=nu, _forget=Default_Lambda, epsilon=Epsilon,resolution=Resolution)
    bayes_m.Fit(X_train)
    return bayes_m

def TrainBayesAda():
    best_nu = arflx.nu
    numDim= len(lags) + 1
    initmu = arflx.param.flatten()
    precision = np.eye(numDim)/10000
    bayesAda_m  = BayesFlx.BayesFlxLogitModel(lags=lags,mu=initmu, precision=precision,nu=best_nu, _forget=Default_Lambda, epsilon=Epsilon,resolution=Resolution)
    bayesAda_m.Fit(X_train)
    return bayesAda_m


class ModelStore:
    def __init__(self):
        self._models = {}
        self._TrainAll()

    def _TrainAll(self):
        self._models["RLS"] = TrainRLS()
        self._models["sRLS"] = TrainSRLS()
        self._models["RMLE"] = TrainRMLE()
        self._models["sRMLE"] = TrainSRMLE()
        self._models["Bayes"] = TrainBayes()
        self._models["BayesAda"] = TrainBayesAda()

    def Get(self, name):
        if name not in self._models: raise KeyError(f"Unknown model {name}")
        return copy.deepcopy(self._models[name])

MODEL_STORE = ModelStore()


###### Disturbation functions ######
def MatrixDisturbe(P):
    numDim = P.shape[0]
    noise_level = np.min([50, np.max(np.linalg.eigvals(P))]) * np.random.rand()
    A = np.random.rand(numDim, numDim) * noise_level
    A = (A + A.T) / 2 # Step 2: Make the matrix symmetric
    return P + A

def NuDisturbe(nu):
    noise_level = nu * 0.05
    noise = noise_level * (np.random.rand() - 0.5) 
    res = nu + noise
    return res

def MuDisturbe(mu):
    numdim = mu.shape[0]
    noise_level = np.min([0.05, np.max(np.abs(mu))])
    noise = noise_level * np.random.rand(numdim)
    res = mu + noise
    return res

def VarzDisturbe(var_z):
    noise_level = var_z * 0.1
    noise = noise_level * (np.random.rand() - 0.5) 
    res = var_z + noise
    return res

###### Metric func ######
def SkillWrapper(model):
    model_pred = model.ProbaPred(X_test)
    model_pred_CRPSs = EvaluateCRPS.NanNumeCRPSs(model_pred,X_test,Epsilon,1-Epsilon,resolution=Resolution)
    res = np.nanmean(model_pred_CRPSs)
    # print(res)
    skill = (Persistence_CRPS- res) / (Persistence_CRPS)
    return skill

###### Processe Wrapper ######
###### RLS
def RLS_mu_Wrapper(i):
    print("RLS_mu", i)
    m = MODEL_STORE.Get("RLS")
    m.mu = MuDisturbe(m.mu)
    skill = SkillWrapper(m)
    return skill

def RLS_Var_Wrapper(i):
    print("RLS_Var", i)
    m = MODEL_STORE.Get("RLS")
    m.beta = VarzDisturbe(m.beta)
    skill = SkillWrapper(m)
    return skill

def RLS_M_Wrapper(i):
    print("RLS_M", i)
    m = MODEL_STORE.Get("RLS")
    m.pR = MatrixDisturbe(m.pR)
    skill = SkillWrapper(m)
    return skill

###### sRLS
def SRLS_mu_Wrapper(i):
    print("sRLS_mu", i)
    m = MODEL_STORE.Get("sRLS")
    m.mu = MuDisturbe(m.mu)
    skill = SkillWrapper(m)
    return skill

def SRLS_Var_Wrapper(i):
    print("sRLS_var", i)
    m = MODEL_STORE.Get("sRLS")
    m.beta = VarzDisturbe(m.beta)
    skill = SkillWrapper(m)
    return skill

def SRLS_M_Wrapper(i):
    print("sRLS_M", i)
    m = MODEL_STORE.Get("sRLS")
    m.P = MatrixDisturbe(m.P)
    skill = SkillWrapper(m)
    return skill

###### RMLE
def RMLE_mu_Wrapper(i):
    print("RMLE_mu", i)
    m = MODEL_STORE.Get("RMLE")
    m.mu = MuDisturbe(m.mu)
    skill = SkillWrapper(m)
    return skill

def RMLE_var_Wrapper(i):
    print("RMLE_var", i)
    m = MODEL_STORE.Get("RMLE")
    m.var_z = VarzDisturbe(m.var_z)
    skill = SkillWrapper(m)
    return skill

def RMLE_M_Wrapper(i):
    print("RMLE_M", i)
    m = MODEL_STORE.Get("RMLE")
    m.R = MatrixDisturbe(m.R)
    skill = SkillWrapper(m)
    return skill

def RMLE_nu_Wrapper(i):
    print("RMLE_nu", i)
    m = MODEL_STORE.Get("RMLE")
    m.nu = NuDisturbe(m.nu)
    skill = SkillWrapper(m)
    return skill

###### sRMLE
def SRMLE_mu_Wrapper(i):
    print("SRMLE_mu", i)
    m = MODEL_STORE.Get("sRMLE")
    m.mu = MuDisturbe(m.mu)
    skill = SkillWrapper(m)
    return skill

def SRMLE_var_Wrapper(i):
    print("SRMLE_var", i)
    m = MODEL_STORE.Get("sRMLE")
    m.var_z = VarzDisturbe(m.var_z)
    skill = SkillWrapper(m)
    return skill

def SRMLE_M_Wrapper(i):
    print("SRMLE_M", i)
    m = MODEL_STORE.Get("sRMLE")
    m.P = MatrixDisturbe(m.P)
    skill = SkillWrapper(m)
    return skill

def SRMLE_nu_Wrapper(i):
    print("SRMLE_nu", i)
    m = MODEL_STORE.Get("sRMLE")
    m.nu = NuDisturbe(m.nu)
    skill = SkillWrapper(m)
    return skill

###### Bayes
def Bayes_mu_Wrapper(i):
    print("Bayes_mu", i)
    m = MODEL_STORE.Get("Bayes")
    m.mu = MuDisturbe(m.mu)
    skill = SkillWrapper(m)
    return skill

def Bayes_var_Wrapper(i):
    print("Bayes_var", i)
    m = MODEL_STORE.Get("Bayes")
    m.beta = VarzDisturbe(m.beta)
    skill = SkillWrapper(m)
    return skill

def Bayes_M_Wrapper(i):
    print("Bayes_M", i)
    m = MODEL_STORE.Get("Bayes")
    m.Precision = MatrixDisturbe(m.Precision)
    skill = SkillWrapper(m)
    return skill

###### BayesAda
def BayesAda_mu_Wrapper(i):
    print("BayesAda_mu", i)
    m = MODEL_STORE.Get("BayesAda")
    m.mu = MuDisturbe(m.mu)
    skill = SkillWrapper(m)
    return skill

def BayesAda_var_Wrapper(i):
    print("BayesAda_var", i)
    m = MODEL_STORE.Get("BayesAda")
    m.beta = VarzDisturbe(m.beta)
    skill = SkillWrapper(m)
    return skill

def BayesAda_P_Wrapper(i):
    print("BayesAda_M", i)
    m = MODEL_STORE.Get("BayesAda")
    m.Precision = MatrixDisturbe(m.Precision)
    skill = SkillWrapper(m)
    return skill

def BayesAda_nu_Wrapper(i):
    print("BayesAda_nu", i)
    m = MODEL_STORE.Get("BayesAda")
    m.nu = NuDisturbe(m.nu)
    skill = SkillWrapper(m)
    return skill


if __name__ == "__main__":
    os.makedirs(output_folder, exist_ok=True)

    runs = np.arange(N_RUN)

    experiments = [
        RLS_mu_Wrapper,
        RLS_Var_Wrapper,
        RLS_M_Wrapper,

        SRLS_mu_Wrapper,
        SRLS_Var_Wrapper,
        SRLS_M_Wrapper,

        RMLE_mu_Wrapper,
        RMLE_var_Wrapper,
        RMLE_M_Wrapper,
        RMLE_nu_Wrapper,

        SRMLE_mu_Wrapper,
        SRMLE_var_Wrapper,
        SRMLE_M_Wrapper,
        SRMLE_nu_Wrapper,

        Bayes_mu_Wrapper,
        Bayes_var_Wrapper,
        Bayes_M_Wrapper,

        BayesAda_mu_Wrapper,
        BayesAda_var_Wrapper,
        BayesAda_P_Wrapper,
        BayesAda_nu_Wrapper,
    ]

    for expm in experiments:
        print(f"Running {expm.__name__} ...")
        t0 = time.time()
        with Pool(processes=8) as pool: results = pool.map(expm, runs)
        duration = time.time() - t0
        print(f"{expm.__name__} finished in {duration:.2f}s")

        df = pd.DataFrame({"res": results})
        out_path = os.path.join(output_folder, f"{expm.__name__}.csv")
        df.to_csv(out_path, index=False)

        print(f"Saved to {out_path}")

### Running script (Windows)
__readme__ ="""
cd <Path to the directory of /src>
python "./Scripts/RELEASE.RunSensitivity.py"
"""