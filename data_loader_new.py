import os
import pickle
import numpy as np
import random
from gensim import corpora
#from gensim.models import KeyedVectors

import pdb

def divide_ids(d, alpha=0.8, beta=0.9):
    num = list(range(0,d))
    random.shuffle(num)
    a = int(d*alpha)
    b = int(d*beta)
    train = num[0:a]
    valid = num[a:b]
    test = num[b:d]
    return train,valid,test

class DataLoader:
    def __init__(self, path='data_new', tn_name='012'):
        self.path = path
        self.tn_name = tn_name
        self.max_len = 32
        with open(os.path.join(self.path, 'data.pkl'), 'rb') as f:
            self.data = pickle.load(f)         
        self.dct = corpora.Dictionary.load(os.path.join(path, 'dict.txt')) #id2token
        self.inv_dct = self.dct.token2id
        with open(os.path.join(self.path, 'w2v.pkl'), 'rb') as f:
            self.w2v = pickle.load(f) #id to w2v
        print("Raw data and dictionary loaded.")
        self.load_training_set()
    
    def gen_training_set(self,types=[0,1,2], alpha=0.8, beta=0.9):
        #types={'Hip-Hop':0, 'Metal':1, 'Country':2, 'Jazz':3, 'Electronic':4, 'R&B':5}
        d = np.array(self.data)
        self.tn_set = np.array([]).reshape(-1,4)
        self.va_set = np.array([]).reshape(-1,4)
        self.tt_set = np.array([]).reshape(-1,4)
        for t in types:
            new_data = d[d[:,1]==t]
            new_data[:,2] = self.padding(data = new_data)
            tn, va, tt = divide_ids(len(new_data), alpha=alpha, beta=beta)
            self.tn_set = np.concatenate((self.tn_set, new_data[tn]), axis=0) #not shuffled
            print(len(self.tn_set))
            self.va_set = np.concatenate((self.va_set, new_data[va]), axis=0)
            print(len(self.va_set))
            self.tt_set = np.concatenate((self.tt_set, new_data[tt]), axis=0)
            print(len(self.tt_set))
        with open(os.path.join(self.path, 'training_'+self.tn_name), 'wb') as f:
            pickle.dump(self.tn_set, f)
        with open(os.path.join(self.path, 'valid_'+self.tn_name), 'wb') as f:
            pickle.dump(self.va_set, f)
        with open(os.path.join(self.path, 'test_'+self.tn_name), 'wb') as f:
            pickle.dump(self.tt_set, f)
        print("Training/Validation/Test data saved.")
    
    def load_training_set(self):
        pth = os.path.join(self.path, 'training_'+self.tn_name)
        if os.path.exists(pth):
            with open(os.path.join(self.path, 'training_'+self.tn_name),'rb') as f:
                self.tn_set = pickle.load(f)
            with open(os.path.join(self.path, 'valid_'+self.tn_name),'rb') as f:
                self.va_set = pickle.load(f)
            with open(os.path.join(self.path, 'test_'+self.tn_name),'rb') as f:
                self.tt_set = pickle.load(f)
            print("Training/Validation/Test data loaded.")
        else:
            print('you need to generate training set using "gen_training_set()".')
    
    def padding(self, data=None):
        if data is None:
            data = self.tn_set
        pad_lyrics = []
        for i in data:
            pad = []
            for j in i[2]:
                temp = [self.inv_dct['UNK']]*self.max_len
                temp[0:len(j)] = j
                pad.append(temp)
            pad_lyrics.append(pad)
        return pad_lyrics
    
    def count_number(self):
        t={}
        for i in self.data:
            try:
                t[i[1]] += 1
            except:
                t[i[1]] = 1
        print (t)
        
        t_train={}
        for i in self.tn_set:
            try:
                t_train[i[1]] += 1
            except:
                t_train[i[1]] = 1
        print (t_train)

        t_val={}
        for i in self.va_set:
            try:
                t_val[i[1]] += 1
            except:
                t_val[i[1]] = 1
        print (t_val)

        t_test={}
        for i in self.tt_set:
            try:
                t_test[i[1]] += 1
            except:
                t_test[i[1]] = 1
        print (t_test)

dd = DataLoader()
pdb.set_trace()
# d.gen_training_set(types=[1,2,5])
# print(dd.count_number())
#types={'Hip-Hop':0, 'Metal':1, 'Country':2, 'Jazz':3, 'Electronic':4, 'R&B':5}
#data[][0]: title; data[][1]: genre; data[][2]: lyrics; data[][3]: length
#title=d.tn_set[:,0]
#w2v[1]


