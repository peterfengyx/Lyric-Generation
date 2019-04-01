# -*- coding: utf-8 -*-
import csv
import re
import os
import pickle
import numpy as np
from gensim import corpora
from gensim.models import KeyedVectors
import string
s=[]
regexp = re.compile(r'[^\x00-\x7f]')

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"won\`t", "will not", phrase)
    phrase = re.sub(r"can\`t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    phrase = re.sub(r"in\'", "ing", phrase)
    
    phrase = re.sub(r"n\`t", " not", phrase)
    phrase = re.sub(r"\`re", " are", phrase)
    phrase = re.sub(r"\`s", " is", phrase)
    phrase = re.sub(r"\`d", " would", phrase)
    phrase = re.sub(r"\`ll", " will", phrase)
    phrase = re.sub(r"\`t", " not", phrase)
    phrase = re.sub(r"\`ve", " have", phrase)
    phrase = re.sub(r"\`m", " am", phrase)
    phrase = re.sub(r"in\`", "ing", phrase)
    
    phrase = re.sub(r"h+ ","h ", phrase)
    phrase = re.sub(r"x+ ","x ", phrase)
    return phrase

r1 = "[\s+-\.\!\/_,$%\?\;\:^*\<\>\(\)+\"\']+|[+——！`，。？、~@#￥%……&*（）]+"
'''
with open(os.path.join('data','w2v.pkl'),'rb') as f:
    w2v=pickle.load(f)
'''
with open(os.path.join('data','raw_data.pkl'),'rb') as f:
    data=pickle.load(f)
for i in range(len(data)-1,-1,-1):
    pp=0
    if len(data[i][5])<30:
        data.pop(i)
        continue
    data[i][5]=re.sub(u"\\(.*?\\)|\\{.*?}|\\[.*?]", "", data[i][5]) #remove parenthesis
    #data[i][5]=re.sub(r'[\d]+', "", data[i][5]) #remove numbers
    data[i][5]=decontracted(data[i][5])
    temp=data[i][5].lower().split('\n')
    for j in range(len(temp)-1,-1,-1):    
        temp[j]=re.sub(r1," ", temp[j]) #remove punctuation
        temp[j]=" ".join(temp[j].split()) #remove redundant blank space
        if len(temp[j])<2:
            temp.pop(j)
            continue
        if len(temp[j].split()>28):
            pp=1
    if pp==1:
        data.pop(i)
        continue
    if len(temp)<4:
        data.pop(i)
        continue
    data[i][5]=temp
with open(os.path.join('data','dataf.pkl'),'wb') as f:
    pickle.dump(data,f)
'''
with open(os.path.join('data','dataf.pkl'),'rb') as f:
    data=pickle.load(f)
'''

document=[]
for i in data:
    for j in i[5]:
        document.append(j.split())
dct=corpora.Dictionary(document)
dct.filter_extremes(no_below=50, keep_n=10000)

#dct=corpora.Dictionary.load('data/dict.txt')
inv_dct=dct.token2id

bad_ids=[]

w2v = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
for i in inv_dct.keys():
    try:
        w2v[i]
    except:
        bad_ids.append(inv_dct[i])
dct.add_documents([["SOS","EOS","UNK"]])
dct.filter_tokens(bad_ids=bad_ids)
dct.save('data/dict.txt')
inv_dct=dct.token2id

id2w2v={}
for i in inv_dct.keys():
    id2w2v[inv_dct[i]]=w2v[i]

with open(os.path.join('data','w2v.pkl'),'wb') as f:
    pickle.dump(w2v,f)

t={}
for i in data:
    try:
        t[i[4]]+=1
    except:
        t[i[4]]=1

types={'Hip-Hop':0, 'Metal':1, 'Country':2, 'Jazz':3, 'Electronic':4, 'R&B':5}
data_toc=[]
for i in data:
    temp=i[1].lower().split('-')
    new=[]
    title=[]
    txt=[]
    length=[]
    for j in temp:
        if j in inv_dct:
            title.append(inv_dct[j])
        else:
            title.append(inv_dct['UNK'])
    new.append(title)
    new.append(types[i[4]])
    for j in i[5]:
        line=[]
        line.append(inv_dct['SOS'])
        temp=j.split(' ')
        for k in temp:
            if k in inv_dct:
                line.append(inv_dct[k])
            else:
                line.append(inv_dct['UNK'])
        line.append(inv_dct['EOS'])
        length.append(len(line))
        txt.append(line)
    new.append(txt)
    new.append(length)
    data_toc.append(new)

with open(os.path.join('data','data.pkl'),'wb') as f:
    pickle.dump(data_toc,f)

mxlen=0
for i in range(len(data_toc)):
    for j in data_toc[i][3]:
        if j>mxlen:
            mxlen=j
            i0=i
#max length is 32

            


    

