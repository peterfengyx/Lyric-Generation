import pdb
from lyric_models import *
import pickle
import numpy as np
from gensim import corpora

import torch.utils.data as data_utils

# load data
train_set = pickle.load(open('data/training_012','rb'))
val_set = pickle.load(open('data/valid_012','rb'))
# load dictionary
idx2word = corpora.Dictionary.load('data/dict.txt')
# load w2v vectors
idx2vec = pickle.load(open('data/w2v.pkl','rb'))
# special token idx
SOS = 9744
EOS = 9743
UNK = 9745
# maximum line length
MaxLineLen = 32 
# maximum lyric length
MaxLyricLen = 40 # Need to be reset

# pdb.set_trace()

class LyricDataset(data_utils.Dataset):
    def __init__(self, lyric_set, max_line_num = MaxLyricLen):
        self.lyric_set = lyric_set
        self.max_line_num = max_line_num
        self.len = len(lyric_set)
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        title = np.mean(np.array([idx2vec[key] for key in self.lyric_set[index][0]]), axis=0)
        genre = self.lyric_set[index][1]
        lyric = self.lyric_set[index][2]
        line_length = self.lyric_set[index][3]


        # while len(caption_numberized) < self.max_length:
        #     caption_numberized.append(word_to_index.get("<UNK>", 0))

        # caption_numberized_array = np.array(caption_numberized)
        # caption_numberized_tensor = torch.from_numpy(caption_numberized_array)

        return {'title': title, 'genre': genre, 'lyric': lyric, 'line_length': line_length}

def trainEpochs(batch_size):
    train_loader = data_utils.DataLoader(dataset=LyricDataset(train_set),
                                         batch_size=batch_size,
                                         shuffle=False)
    val_loader = data_utils.DataLoader(dataset=LyricDataset(val_set),
                                       batch_size=batch_size,
                                       shuffle=True)
    
    for batch, data in enumerate(train_loader, 0):
            title_tensor = data['title']
            # image_feature_variable = Variable(image_feature_tensor)
            genre_tensor = data['genre']
            # caption_numberized_variable = Variable(caption_numberized_tensor)
            lyric_tensor = data['lyric']

            line_length_tensor = data['line_length']

            pdb.set_trace()

if __name__=='__main__':
    batch_size = 10
    trainEpochs(batch_size)


#     sentence_encoder = SentenceEncoder()
#     sentence_encoder = cudalize(sentence_encoder)
#     sentence_encoder.train()
