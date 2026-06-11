import numpy as np

def NanSkill(benchmark:np.array, reference: np.array, idea = 0):
    return  (benchmark - reference) / (benchmark - idea)

def CleanSkill(benchmark:np.array, reference: np.array, idea = 0):
    nanskill = NanSkill(benchmark,reference,idea)
    cleanskill = nanskill[~np.isnan(nanskill)]
    return cleanskill
