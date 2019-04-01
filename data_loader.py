import csv
import re
import os
import pickle
import numpy as np
from gensim import corpora
#from gensim.models import KeyedVectors
import string


class DataLoader:
    def __init__(self,path='data'):
        self.path=path
        with open(os.path.join(self.path,'data.pkl'),'rb') as f:
            self.data=pickle.load(f) 
        self.dct=corpora.Dictionary.load(os.path.join(path,'dict.txt')) #id2token
        self.inv_dct=dct.token2id
        with open(os.path.join(self.path,'w2v.pkl'),'rb') as f:
            self.w2v=pickle.load(f) #id to w2v


d=DataLoader()   


