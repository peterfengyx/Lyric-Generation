import pdb
from lyric_models import *
import pickle
import numpy as np
from gensim import corpora
from tensorboardX import SummaryWriter

import torch.utils.data as data_utils
import sys
import os

# input from command line
if len(sys.argv) != 3:
    raise ValueError("Wrong argument number!")

BatchSize = int(sys.argv[1]) # 20
SavingDir = sys.argv[2]
LearningRate = 0.0001
print ('BatchSize: ', BatchSize, ' SavingDir: ', SavingDir)

if not os.path.exists(SavingDir):
    os.makedirs(SavingDir)
# --------------------------- Load Data ---------------------------
train_set = pickle.load(open('data_new/training_012','rb'))
val_set = pickle.load(open('data_new/valid_012','rb'))
#--------------------------- Meta Data ---------------------------
# special token idx
SOS = 9744
EOS = 9743
UNK = 9745
# maximum line length
MaxLineLen = 32 
# maximum lyric length
MaxLineNum = 40 # Need to be reset
# dictionary size
DictionarySize = 9746
# genre size
GenreSize = 3
TitleSize = 300
#----------------------------------------------------------------
# load dictionary
# idx2word = corpora.Dictionary.load('data_new/dict.txt')
# load w2v vectors
# idx2vec = pickle.load(open('data_new/w2v.pkl','rb'))
word_embedding = np.eye(DictionarySize)
title_embedding = pickle.load(open('data_new/w2v_embedding.pkl','rb'))
# genre_embedding = torch.eye(GenreSize)
# line_end_embedding = torch.eye(MaxLineNum).type(torch.LongTensor)

writer = SummaryWriter()
#----------------------------------------------------------------
class LyricDataset(data_utils.Dataset):
    def __init__(self, lyric_set, max_line_num = MaxLineNum):
        self.lyric_set = lyric_set
        self.max_line_num = max_line_num
        self.len = len(lyric_set)
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        title = np.mean(np.array([title_embedding[key] for key in self.lyric_set[index][0]]), axis=0)
        genre = self.lyric_set[index][1]
        lyric = self.lyric_set[index][2]
        line_length = self.lyric_set[index][3]

        line_numb = len(lyric)
        if line_numb > self.max_line_num:
            lyric = lyric[:self.max_line_num]
            line_length = line_length[:self.max_line_num]
            line_numb = self.max_line_num
        else:
            for _ in range(self.max_line_num - line_numb):
                lyric.append([UNK]*MaxLineLen)
                line_length.append(0)
        
        return {'title': title, 'genre': genre, 'lyric': np.array(lyric), 'line_length': np.array(line_length), 'line_numb': line_numb}

def train_val(model_type,
              genre_tensor,
              lyric_tensor,
              line_length_tensor,
              line_num_tensor,
              sentence_encoder,
              lyric_encoder,
              lyric_classifier,
              sentence_encoder_optimizer,
              lyric_encoder_optimizer,
              lyric_classifier_optimizer,
              cross_entropy_loss,
              batch_size):
     
    if model_type == 'train':
        sentence_encoder_optimizer.zero_grad()
        lyric_encoder_optimizer.zero_grad()
        lyric_classifier_optimizer.zero_grad()
    
    class_loss_data = 0.0

    line_number = torch.max(line_num_tensor).item()
    line_length = torch.max(line_length_tensor).item()

    # pdb.set_trace()
    le_hidden = cudalize(Variable(lyric_encoder.initHidden(batch_size))) # torch.Size([1, 10, 512])
    le_hiddens_variable = le_hidden # torch.Size([1, 10, 512])
    
    for line_num in range(line_number):
        se_hidden = cudalize(Variable(sentence_encoder.initHidden(batch_size))) # torch.Size([1, 10, 512])
        se_hiddens_variable = se_hidden # torch.Size([1, 10, 512])
        
        for line_idx in range(line_length):
            se_input = torch.from_numpy(word_embedding[lyric_tensor[:,line_num,line_idx]]).type(torch.FloatTensor) # torch.Size([10, 9746])
            se_input = cudalize(Variable(se_input))
            _, se_hidden = sentence_encoder(se_input, se_hidden, batch_size)
            se_hiddens_variable = torch.cat((se_hiddens_variable, se_hidden))

        le_input = se_hiddens_variable[line_length_tensor[:,line_num], np.arange(batch_size), :] # torch.Size([10, 512])

        _, le_hidden = lyric_encoder(le_input, le_hidden, batch_size)
        le_hiddens_variable = torch.cat((le_hiddens_variable, le_hidden))
    
    lyric_un_latent_variable = le_hiddens_variable[line_num_tensor, np.arange(batch_size), :] # torch.Size([10, 512])

    predicted_genre = lyric_classifier(lyric_un_latent_variable)
    class_loss = cross_entropy_loss(predicted_genre, cudalize(genre_tensor))
    class_loss_data = class_loss.item()

    if model_type == 'train':
        class_loss.backward()

        sentence_encoder_optimizer.step()
        lyric_encoder_optimizer.step()
        lyric_classifier_optimizer.step()
    
    return class_loss_data

def trainEpochs(sentence_encoder, 
                lyric_encoder, 
                lyric_classifier, 
                batch_size, 
                learning_rate, 
                num_epoch, 
                print_every,
                saving_dir = SavingDir):
    sentence_encoder_optimizer = torch.optim.Adam(sentence_encoder.parameters(), lr=learning_rate)
    lyric_encoder_optimizer = torch.optim.Adam(lyric_encoder.parameters(), lr=learning_rate)
    lyric_classifier_optimizer = torch.optim.Adam(lyric_classifier.parameters(), lr=learning_rate)

    train_loader = data_utils.DataLoader(dataset=LyricDataset(train_set),
                                         batch_size=batch_size,
                                         shuffle=True) # True)
    val_loader = data_utils.DataLoader(dataset=LyricDataset(val_set),
                                       batch_size=batch_size,
                                       shuffle=True)
    cross_entropy_loss = nn.CrossEntropyLoss()

    iter_epoch = 0
    for epoch in range(num_epoch):
        sentence_encoder.train()
        lyric_encoder.train()
        lyric_classifier.train()

        print_loss_total_class = 0.0  # Reset every print_every
        print_loss_total_class_list = []

        for batch, data in enumerate(train_loader, 0):
            # title_tensor = data['title'].type(torch.FloatTensor) # torch.Size([10, 9746])
            genre_tensor = data['genre'] # torch.Size([10]), tensor([0, 2, 1, 1, 0, 1, 2, 1, 1, 1])
            lyric_tensor = data['lyric'] # torch.Size([10, 40, 32])
            line_length_tensor = data['line_length'] # torch.Size([10, 40])
            line_num_tensor = data['line_numb'] # torch.Size([10]), tensor([40, 17, 31, 38, 40, 40, 22,  9, 12, 39])

            # print(batch)
            class_loss = train_val('train',
                                   genre_tensor,
                                   lyric_tensor,
                                   line_length_tensor,
                                   line_num_tensor,
                                   sentence_encoder,
                                   lyric_encoder,
                                   lyric_classifier,
                                   sentence_encoder_optimizer,
                                   lyric_encoder_optimizer,
                                   lyric_classifier_optimizer,
                                   cross_entropy_loss,
                                   len(line_num_tensor))
        
            print_loss_total_class += class_loss
            print_loss_total_class_list.append(class_loss)

            if batch % print_every == (print_every-1):
                print_loss_avg_class = print_loss_total_class / print_every
                print_loss_total_class = 0.0

                print('[%d, %d]  [%.6f]' % (epoch+1, batch+1, print_loss_avg_class))
            
        print_loss_class_avg_train = np.mean(np.array(print_loss_total_class_list))

        print('Train loss: [%.6f, %.6f, %6f]' % (print_loss_class_avg_train))

        # validation
        sentence_encoder.eval()
        lyric_encoder.eval()
        lyric_classifier.eval()
        
        validation_loss_class_list = []

        for _, val_data in enumerate(val_loader, 0):
            # title_tensor = val_data['title'].type(torch.FloatTensor) # torch.Size([10, 9746])
            genre_tensor = val_data['genre'] # torch.Size([10]), tensor([0, 2, 1, 1, 0, 1, 2, 1, 1, 1])
            lyric_tensor = val_data['lyric'] # torch.Size([10, 40, 32])
            line_length_tensor = val_data['line_length'] # torch.Size([10, 40])
            line_num_tensor = val_data['line_numb'] # torch.Size([10]), tensor([40, 17, 31, 38, 40, 40, 22,  9, 12, 39])

            class_loss = train_val('val',
                                   genre_tensor,
                                   lyric_tensor,
                                   line_length_tensor,
                                   line_num_tensor,
                                   sentence_encoder,
                                   lyric_encoder,
                                   lyric_classifier,
                                   sentence_encoder_optimizer,
                                   lyric_encoder_optimizer,
                                   lyric_classifier_optimizer,
                                   cross_entropy_loss,
                                   len(line_num_tensor))

            validation_loss_class_list.append(class_loss)
        
        print_loss_class_avg_val = np.mean(np.array(validation_loss_class_list))

        print('        Validation loss: [%.6f]' % (print_loss_class_avg_val))
        
        # write to tensorboard
        iter_epoch += 1
        writer.add_scalars(saving_dir+'/class_loss/train_val_epoch', {'train': print_loss_class_avg_train, 'val': print_loss_class_avg_val}, iter_epoch)
        
        # # save models    
        # torch.save(sentence_encoder.state_dict(), saving_dir+'/sentence_encoder_'+str(epoch+1))
        # torch.save(lyric_encoder.state_dict(), saving_dir+'/lyric_encoder_'+str(epoch+1))
        # torch.save(lyric_classifier.state_dict(), saving_dir+'/lyric_generator_'+str(epoch+1))
        
if __name__=='__main__':
    word_embedding_size = DictionarySize
    title_embedding_size = TitleSize
    genre_embedding_size = GenreSize

    # sentence encoder - se
    se_input_size = word_embedding_size #  + title_embedding_size + genre_embedding_size
    se_embedding_size = 128
    se_hidden_size = 128 # 512
    sentence_encoder = SentenceEncoder(se_input_size, se_embedding_size, se_hidden_size)
    sentence_encoder = cudalize(sentence_encoder)
    sentence_encoder.train()

    # lyric encoder - le
    le_input_size = se_hidden_size # + title_embedding_size + genre_embedding_size
    le_embedding_size = 128 # not used
    le_hidden_size = 128
    lyric_encoder = LyricEncoder(le_input_size, le_embedding_size, le_hidden_size)
    lyric_encoder = cudalize(lyric_encoder)
    lyric_encoder.train()

    # lyric classifier - lc
    lc_input_size = le_hidden_size
    lc_output_size = GenreSize
    lyric_classifier = LyricClassifier(lc_input_size, lc_output_size)
    lyric_classifier = cudalize(lyric_classifier)
    lyric_classifier.train()

    batch_size = BatchSize # 20 # 20
    learning_rate = LearningRate
    num_epoch = 5000
    print_every = 1
    
    trainEpochs(sentence_encoder, lyric_encoder, lyric_classifier, batch_size, learning_rate, num_epoch, print_every)
    writer.close()
