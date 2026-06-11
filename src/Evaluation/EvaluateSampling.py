import numpy as np

def BootstrapSampling(samples, size, merge_func=np.average):
    """
    Use Bootstrap Sampling Method to get distribution of evaluation metric (CRPS or Skill Score or any else)
    """
    has_nan = np.any(np.isnan(samples))
    if has_nan: raise ValueError("should not has nan in samples")
    repeated = np.random.choice(samples, (len(samples),size))
    merged = np.apply_along_axis(merge_func,0,repeated)
    return merged