import pandas as pd
import numpy as np
import datetime as dt

def IsStartSummerTime(datetime:dt.datetime)-> bool:
    """
    Detect the 1 o'clock (UTC+0) of the last sunday of March 
    """
    if not datetime.month == 3: return False
    is_Next_week_April =  (datetime + dt.timedelta(days=7)).month == 4
    if not is_Next_week_April: return False
    if not datetime.weekday() == 6 : return False # because 0 is Monday
    if not datetime.hour == 1 : return False
    return True

def IsEndSummerTime(datetime:dt.datetime)->bool:
    """
    Detect the 1 o'clock (UTC+0) of the last sunday of Octobre 
    """
    if not datetime.month == 10: return False
    is_Next_week_Nov =  (datetime + dt.timedelta(days=7)).month == 11
    if not is_Next_week_Nov: return False
    if not datetime.weekday() == 6 : return False # because 0 is Monday
    if not datetime.hour == 1 : return False
    return True

def GetTimeIndex(year:int, output_name="", shred_per_hour = 2 )-> pd.DataFrame:
    """
    Generate dtm_utc, sp_utc, dtm_local, sd(str), sp as benchmark index
    
    year: int.
    output_name: string. if non-empty string, generate a .csv with given name. skip output file by given empty string (default).
    shred_per_hour: int. should devide 60.
    
    dtm_utc: <M8[ns], datetime
    sp_utc : int, counter number of time shred of a day
    dtm_local: <M8[ns], datetime, considering summer daylight saving time
    sd     : string of date yyyy-mm-dd, considering summer daylight saving time
    sp     : int, counter number of thime shred of a day, considering summer daylight saving time
    """
    
    next_year = year + 1
    total_days = (dt.datetime(next_year, 1, 1) - dt.datetime(year,1,1)).days
    total_shred = total_days * 24 * shred_per_hour
    shred_per_day = 24 * shred_per_hour
    time_step_size = dt.timedelta(minutes = (int)(60/shred_per_hour))
    
    sp_utc = (np.linspace(1,total_shred, total_shred, endpoint=True, dtype = 'int') - 1 ) % shred_per_day + 1
    dtm_utc = np.array([dt.datetime(year,1,1)] * total_shred)
    dtm_local = np.array([dt.datetime(year,1,1)] * total_shred)
    sd =  np.array([dt.datetime(year,1,1).strftime("%Y-%m-%d")] * total_shred)
    sp = np.zeros(total_shred, dtype='int')
    current_datetime = dt.datetime(year,1,1) - time_step_size
    summer_time = False
    summer_time_shred_saved = False # start with winter time
    
    for step in range(total_shred):
        current_datetime += time_step_size
        dtm_utc[step] = current_datetime
        
        if(IsStartSummerTime(current_datetime)) : summer_time = True
        if(IsEndSummerTime(current_datetime)) : summer_time = False
            
        if summer_time: 
            sd_datetime = current_datetime + dt.timedelta(hours=1)
            crt_raw_sp = sp_utc[step]  + 2 # try save 2 time shred
            if not summer_time_shred_saved: 
                if crt_raw_sp > 48: summer_time_shred_saved = True
                else: crt_raw_sp -= 2 
            dtm_local[step] = sd_datetime
            sd[step] = sd_datetime.strftime("%Y-%m-%d")
            sp[step] = (crt_raw_sp - 1 ) % 48 + 1   
            
        else: # winter time
            sd_datetime = current_datetime 
            crt_raw_sp = sp_utc[step]
            if summer_time_shred_saved:
                crt_raw_sp += 2
                if crt_raw_sp >= 50: summer_time_shred_saved = False
            dtm_local[step] = sd_datetime
            sd[step] = sd_datetime.strftime("%Y-%m-%d")
            sp[step] = crt_raw_sp

    df = pd.DataFrame({"dtm":dtm_utc, "sp_utc":sp_utc, "dtm_local":dtm_local, "sd":sd,"sp":sp})
    if(len(output_name)): df.to_csv(f"{output_name}.csv")
    return df

def DetectDuplicatedSdSp(df:pd.DataFrame,on=["sd","sp"]):
    duplication = df.duplicated(on)
    return np.sum(duplication), df[duplication]

def DetectMissingSdSp(supper:pd.DataFrame, subset:pd.DataFrame, on=["sd","sp"])-> list:
    """
    Detect and Print any 
    """
    key0 = on[0]
    key1 = on[1]
    supper_set = set(list(zip(supper[key0],supper[key1])))
    subset_set = set(list(zip(subset[key0],subset[key1])))
    return list(supper_set.difference(subset_set))
