import os
import pandas as pd


import _bootstrap
import Configuration
Output_Folder = Configuration.OUTPUT_FOLDER
Data_Folder = Configuration.DATA_FOLDER


import Domain.Constant as Constant
import DataCleaning.DataQualities as DQ


###### Meta Input ######
Data_Years = [2020,2021,2022,2023]


########################
if(not os.path.exists(os.path.join(Data_Folder,"WindFarmData","DQ_Summary.csv"))):
    print("initialise a UBRDQ csv")
    UBRDQ_df = pd.DataFrame([{'Windfarm': 'Speciman','Year': 0,'SpikeThresholdConfig': 0.995,'EpsilonConfig': 0.005,'MaxSpikePcentConfig': 10,
                              'NanCount':0, 'NanPcent':0,'ZeroCount': 0,'ZeroPcent': 0.0,'SpikeCount': 0,'SpikeCPcent': 0.0,'EplisonCount': 0,'EplisonPcent': 0.0,
                              'CleanMax': 0.0,'CleanEpsilonMaxCount': 0,'CleanEpsilonMaxPcent': 0.0}])

for year in Data_Years:
    UBR_file_path = os.path.join(Data_Folder, f"UBOR_{year}.csv")
    if(not os.path.exists(UBR_file_path)):raise AssertionError(f"{UBR_file_path} not exists")
    UBOR_df = pd.read_csv(UBR_file_path)
    windfarms = list(set(Constant.WINDFARMS).intersection(set(UBOR_df.columns)))

    for windfarm in windfarms:
        print(end = "\r")
        print(f"processing {year} {windfarm}",end = "\r")
        dq, dqdict = DQ.DataQualitySummarise(UBOR_df[windfarm],windfarm,year,strict_continuous=True)
        UBRDQ_df.loc[len(UBRDQ_df)] = dqdict
        cleandata = UBOR_df[windfarm].dropna()


UBRDQ_df = UBRDQ_df[~UBRDQ_df.duplicated(subset=["Windfarm","Year"],keep='last')]
UBRDQ_df.to_csv(os.path.join(Output_Folder,"DQ_Summary.csv"), index=False)
