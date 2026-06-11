import numpy as np
from typing import List
from DataCleaning import ResultPostProcess as ResPro
from Models import IModel
__DEFAULT_CONFIG = {
  "Epsilon": 0.005,
  "Resolution" : 501

}

class ModelFactory():
  __Config : dict
  def __init__(self, config):
    self.__Config = config
    self.__CreatorMap = {
    "persistent": self.CreatePersistent

  }
    pass
  
  def Create(self, modelname):
    lowername = str.lower(modelname)
    if lowername in self.__CreatorMap.keys:
      return self.__CreatorMap[lowername]()
    else: return None

  def CreatePersistent(self)->IModel.IDBdSeqPrbPredModel:
    pass

  