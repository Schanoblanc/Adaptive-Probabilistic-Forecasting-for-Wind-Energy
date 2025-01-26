import os
import sys
import pandas as pd
import numpy as np
from src.release import *

Root_Folder = os.getcwd()
print(Root_Folder)
if Root_Folder not in sys.path: sys.path.append(Root_Folder)

def RunModelWrapper(X_train, X_test):
    lags = SelectOrder(X_train)
    persistence_model = PersistenceModel(epsilon=Epsilon, resolution=Resolution)
    persistence_model.Fit(X_train)
    persistence_pred = persistence_model.ProbaPred(X_test)
    persistence_CRPSs = NanNumeCRPSs(persistence_pred, X_test, Epsilon, 1 - Epsilon, resolution=Resolution)
    persistence_CRPS = np.nanmean(persistence_CRPSs)

    ar_model = ARModel(lags=lags, nu=1, epsilon=Epsilon, resolution=Resolution)
    ar_model.Fit(X_train)
    ar_pred = ar_model.ProbaPred(X_test)
    ar_CRPSs = NanNumeCRPSs(ar_pred, X_test, Epsilon, 1 - Epsilon, resolution=Resolution)
    ar_CRPS = np.nanmean(ar_CRPSs)

    ar_nu_model = ARnuModel(lags=lags, inital_nu=1, epsilon=Epsilon, resolution=Resolution)
    ar_nu_model.Fit(X_train, nv_bounds=[0.1, 3])
    ar_nu_pred = ar_nu_model.ProbaPred(X_test)
    ar_nu_CRPSs = NanNumeCRPSs(ar_nu_pred, X_test, Epsilon, 1 - Epsilon, resolution=Resolution)
    ar_nu_CRPS = np.nanmean(ar_nu_CRPSs)

    nu = ar_nu_model.nu
    initmu = ar_nu_model.param.flatten()
    numDim = len(lags) + 1
    precision = np.eye(numDim)
    beta = ar_nu_model.sigma**2
    rls_model = RLSModel(lags, initmu, precision, beta, nu)
    rls_model.Fit(X_train)
    rls_pred = rls_model.ProbaPred(X_test)
    rls_CRPSs = NanNumeCRPSs(rls_pred, X_test, Epsilon, 1 - Epsilon, resolution=Resolution)
    rls_CRPS = np.nanmean(rls_CRPSs)

    mu = ar_nu_model.param.flatten()
    var_z = ar_nu_model.sigma**2
    R = np.eye(len(lags)+3)
    nw_model = NWModel(lags=lags,mu=mu,var_z=var_z,R=R)
    nw_model.Fit(X_train)
    nw_pred = nw_model.ProbaPred(X_test)
    nw_CRPSs = NanNumeCRPSs(nw_pred, X_test, Epsilon, 1 - Epsilon, resolution=Resolution)
    nw_CRPS = np.nanmean(nw_CRPSs)

    nu = ar_nu_model.nu
    initmu = ar_nu_model.param.flatten()
    numDim= len(lags) + 1
    precision = np.eye(numDim)/10000
    bayes_model = BayesModel(lags=lags,mu=initmu, precision=precision,nu=nu, epsilon=Epsilon,resolution=Resolution)
    bayes_model.Fit(X_train)
    bayes_pred = bayes_model.ProbaPred(X_test)
    bayes_CRPSs = NanNumeCRPSs(bayes_pred, X_test, Epsilon, 1 - Epsilon, resolution=Resolution)
    bayes_CRPS = np.nanmean(bayes_CRPSs)
    

    best_nu = ar_nu_model.nu
    numDim = len(lags) + 1
    initmu = ar_nu_model.param.flatten()
    precision = np.eye(numDim) / 10000
    bayes_nu_model = BayesNuModel(lags=lags, mu=initmu, precision=precision, nu=best_nu, epsilon=Epsilon, resolution=Resolution)
    bayes_nu_model.Fit(X_train)
    bayes_nu_pred = bayes_nu_model.ProbaPred(X_test)
    bayes_nu_CRPSs = NanNumeCRPSs(bayes_nu_pred, X_test, Epsilon, 1 - Epsilon, resolution=Resolution)
    bayes_nu_CRPS = np.nanmean(bayes_nu_CRPSs)

    log = f"CRPS:     Persistence:{persistence_CRPS:.5f},\
    AR:{ar_CRPS:.5f},\
    ARnu:{ar_nu_CRPS:.5f},\
    RLS:{rls_CRPS:.5f},\
    NW:{nw_CRPS:.5f},\
    Bayes:{bayes_CRPS:.5f},\
    BayesNu:{bayes_nu_CRPS:.5f}"
    
    print(log)
    return



if "__main__" == __name__:

    windfarm = '2__PSTAT001'
    train_year = 2020
    train_data_file_path = os.path.join(Root_Folder, f"data\\UBOR_{train_year}.csv")
    test_data_file_path = os.path.join(Root_Folder, f"data\\UBOR_{train_year+1}.csv")
    X_train,X_test = GetData(windfarm,train_data_file_path,test_data_file_path)
    RunModelWrapper(X_train, X_test)