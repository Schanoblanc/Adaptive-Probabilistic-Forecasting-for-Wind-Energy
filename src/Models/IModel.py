from abc import ABC, abstractmethod
from numpy import array as nparray
from DataObjectModels.ProbPredResultDO import IProbaPredResult

class IDBdSeqPtPredModel(ABC):
    '''
    Double-Bounded Sequence, Point Prediction Model.
    IMPORTANT: Pred SHOULD/MUST be in SAME doamin, whatever furthing transformation inside model.
    IMPORTANT: Pred SHOULD/MUST have SAME data length. Can fill nan when not able to predict.
    '''
    @abstractmethod 
    def Fit(self, X:nparray)-> None: ...
        # Take Single Sequences as input.(multi sequences?) 

    @abstractmethod 
    def PointPred(self, X:nparray)-> nparray: ...
        # Provided point prediction. output as array
    
    @abstractmethod 
    def PointPredBounded(self, X:nparray, left, right)-> nparray: ...
        # Provided point prediction bounded by left and right. output as array
    
    
class IDBdSeqPrbPredModel(ABC):
    '''
    Double-Bounded Sequence, Probabilistic Prediction Model.
    IMPORTANT: Pred SHOULD/MUST be in SAME doamin, whatever furthing transformation inside model.
    '''
    @abstractmethod 
    def Fit(self, X:nparray)-> None: ...
        # Take Single Sequences as input.(multi sequences?) 

    @abstractmethod 
    def ProbaPred(self, X:nparray)-> IProbaPredResult: ...
        # Provided point prediction. output as array
    

