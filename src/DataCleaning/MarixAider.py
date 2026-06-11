from numpy import ndarray

class MatrixAider:
    _data: ndarray

    def __init__(self):
        pass
    
    def Load(self, X:ndarray):
        self._data = X

    def GetData(self):
        return self._data.copy()