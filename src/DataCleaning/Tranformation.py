import numpy as np

def LogitNormal(x:np.ndarray,nu:float):
    xnu = x ** nu
    res = np.log(xnu/ (1-xnu))
    return res
    
def InverseLogitNormal(y:np.array, nu:float):
    ey = np.e ** y
    return (ey/(1+ey)) ** (1.0/nu)

def SeqToOffsetLagMatrix(x:np.ndarray|list[float], lags:list[int]):
    """
    Given a 1D sequence x_t , and lags l1..ln, Generate matrix whose i-th row is:
    [1, X_i-l1, X_i-l2, ... , X_i-ln].  
    set np.nan when X_i-lj not exists. 
    """
    length = len(x)
    matrix = np.full((len(x),len(lags)+1),np.nan)
    matrix[:,0] =1.0
    for i,lag in enumerate(lags):
        if lag >= length: continue
        matrix[lag:,i+1] = x[:-lag]
    return matrix

def MatrixDropNanRow(matrix:np.ndarray):
    return matrix[~np.isnan(matrix).any(axis=1), :]

def XYDropNaN(X:np.ndarray,Y:np.ndarray):
    assert X.shape[0] == Y.shape[0], "X Y should have same data count"
    XY = np.concatenate((X,Y),axis=1)
    XY= MatrixDropNanRow(XY)
    X = XY[:,0:X.shape[1]]
    Y = XY[:,X.shape[1]:]
    return X,Y

def XPrecision(X:np.ndarray, lags):
    """
    Given lags such that n_dim = len(lags) + 1. Calculate Precision Matrix shape (n_dim, n_dim).
    N.B.: Some peole call this matrix as precision. 
    """
    X_m = SeqToOffsetLagMatrix(X, lags) #shape (N,d)
    X_m = MatrixDropNanRow(X_m)
    Pcsn = X_m.T @ X_m
    return Pcsn