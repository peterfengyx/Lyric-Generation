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
# idx2word = corpora.Dictionary.load('data/dict.txt')
# load w2v vectors
# idx2vec = pickle.load(open('data/w2v.pkl','rb'))
word_embedding = pickle.load(open('w2v_embedding.pkl','rb'))
genre_embedding = torch.eye(3)
pdb.set_trace()
# special token idx
SOS = 9744
EOS = 9743
UNK = 9745
# maximum line length
MaxLineLen = 32 
# maximum lyric length
MaxLineNum = 40 # Need to be reset

# pdb.set_trace()

class LyricDataset(data_utils.Dataset):
    def __init__(self, lyric_set, max_line_num = MaxLineNum):
        self.lyric_set = lyric_set
        self.max_line_num = max_line_num
        self.len = len(lyric_set)
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        title = np.mean(np.array([word_embedding[key] for key in self.lyric_set[index][0]]), axis=0)
        genre = self.lyric_set[index][1]
        lyric = self.lyric_set[index][2]
        line_length = self.lyric_set[index][3]

        line_num = len(lyric)
        if line_num > self.max_line_num:
            lyric = lyric[:self.max_line_num]
            line_length = line_length[:self.max_line_num]
            line_num = self.max_line_num
        else:
            for _ in range(self.max_line_num - line_num):
                lyric.append([UNK]*MaxLineLen)
                line_length.append(0)
        
        return {'title': title, 'genre': genre, 'lyric': np.array(lyric), 'line_length': np.array(line_length), 'line_num': line_num}

def train_val(model_type,
              title_tensor,
              genre_tensor,
              lyric_tensor,
              line_length_tensor,
              line_num_tensor,
              sentence_encoder, # se
              lyric_encoder, # le
              lyric_generator, # lg
              sentence_generator, # sg
              sentence_encoder_optimizer,
              lyric_encoder_optimizer,
              lyric_generator_optimizer,
              sentence_generator_optimizer,
              batch_size,
              max_line_num = MaxLineNum,
              max_line_length = MaxLineLen):
     
    if model_type == 'train':
        sentence_encoder_optimizer.zero_grad()
        lyric_encoder_optimizer.zero_grad()
        lyric_generator_optimizer.zero_grad()
        sentence_generator_optimizer.zero_grad()
    
    auto_loss_data = 0.0

    line_number = torch.max(line_num_tensor).item()
    line_length = torch.max(line_length_tensor).item()

    le_hidden = cudalize(Variable(lyric_encoder.initHidden(batch_size)))
    le_hiddens_variable = le_hidden[0]
    
    for line_num in range(line_number):
        se_hidden = cudalize(Variable(sentence_encoder.initHidden(batch_size)))
        se_hiddens_variable = se_hidden[0]
        
        for line_idx in range(line_length):
            input_word_tensor = torch.from_numpy(word_embedding[lyric_tensor[:,line_num,line_idx].tolist()]).type(torch.FloatTensor)
            # title_tensor - this line
            input_genere_tensor = genre_embedding[genre_tensor]
            se_input = torch.cat((input_word_tensor, title_tensor, input_genere_tensor), 1).view(1, batch_size, -1)
            se_input = cudalize(Variable(se_input))
            _, se_hidden = sentence_encoder(se_input, se_hidden)
            se_hiddens_variable = torch.cat((se_hiddens_variable, se_hidden))
        
        le_input = se_hiddens_variable[line_length_tensor[:,line_num], np.arange(batch_size), :]
        _, le_hidden = sentence_encoder(le_input, le_hidden)
        le_hiddens_variable = torch.cat((le_hiddens_variable, le_hidden))
    lyric_latent = le_hiddens_variable[line_num_tensor, np.arange(batch_size), :]

    # need to do decoder on lyric_latent


    return auto_loss_data

def trainEpochs(sentence_encoder, 
                lyric_encoder, 
                lyric_generator, 
                sentence_generator, 
                batch_size, 
                learning_rate, 
                num_epoch, 
                print_every):
    sentence_encoder_optimizer = torch.optim.Adam(sentence_encoder.parameters(), lr=learning_rate)
    lyric_encoder_optimizer = torch.optim.Adam(lyric_encoder.parameters(), lr=learning_rate)
    lyric_generator_optimizer = torch.optim.Adam(lyric_generator.parameters(), lr=learning_rate)
    sentence_generator_optimizer = torch.optim.Adam(sentence_generator.parameters(), lr=learning_rate)

    train_loader = data_utils.DataLoader(dataset=LyricDataset(train_set),
                                         batch_size=batch_size,
                                         shuffle=True)
    val_loader = data_utils.DataLoader(dataset=LyricDataset(val_set),
                                       batch_size=batch_size,
                                       shuffle=True)
    iter_epoch = 0
    for epoch in range(num_epoch):
        sentence_encoder.train()
        lyric_encoder.train()
        lyric_generator.train()
        sentence_generator.train()

        for batch, data in enumerate(train_loader, 0):
            title_tensor = data['title']
            # image_feature_variable = Variable(image_feature_tensor)
            genre_tensor = data['genre']
            # caption_numberized_variable = Variable(caption_numberized_tensor)
            lyric_tensor = data['lyric']
            line_length_tensor = data['line_length']
            line_num_tensor = data['line_num']

            pdb.set_trace()

            loss = train_val('train',
                             title_tensor,
                             genre_tensor,
                             lyric_tensor,
                             line_length_tensor,
                             line_num_tensor,
                             sentence_encoder,
                             lyric_encoder,
                             lyric_generator,
                             sentence_generator,
                             sentence_encoder_optimizer,
                             lyric_encoder_optimizer,
                             lyric_generator_optimizer,
                             sentence_generator_optimizer,
                             len(line_num_tensor))
        
        # validation
        sentence_encoder.eval()
        lyric_encoder.eval()
        lyric_generator.eval()
        sentence_generator.eval()
        # need to complete

        # write to tensorboard
        iter_epoch += 1
        # need to complete

        # save models    
        torch.save(sentence_encoder.state_dict(), saving_dir+'/sentence_encoder_'+str(epoch+1))
        torch.save(lyric_encoder.state_dict(), saving_dir+'/lyric_encoder_'+str(epoch+1))
        torch.save(lyric_generator.state_dict(), saving_dir+'/lyric_generator_'+str(epoch+1))
        torch.save(sentence_generator.state_dict(), saving_dir+'/sentence_generator_'+str(epoch+1))

if __name__=='__main__':
    vocabulary_size = len(word_embedding)

    word_embedding_size = 300
    title_embedding_size = 300
    genre_embedding_size = 3

    # sentence encoder - se
    se_input_size = word_embedding_size + title_embedding_size + genre_embedding_size
    se_hidden_size = 512
    sentence_encoder = SentenceEncoder(se_input_size, se_hidden_size)
    sentence_encoder = cudalize(sentence_encoder)
    sentence_encoder.train()

    # lyric encoder - le
    le_input_size = se_hidden_size
    le_hidden_size = 512
    lyric_encoder = LyricEncoder(le_input_size, le_hidden_size)
    lyric_encoder = cudalize(lyric_encoder)
    lyric_encoder.train()

    # lyric generator - lg
    lg_input_size = le_hidden_size + title_embedding_size + genre_embedding_size
    lg_hidden_size = 512
    lg_topic_latent_size = 512
    lg_topic_output_size = 512
    lyric_generator = LyricGenerator(lg_input_size, lg_hidden_size, lg_topic_latent_size, lg_topic_output_size)
    lyric_generator = cudalize(lyric_generator)
    lyric_generator.train()

    # sentence generator - sg
    sg_input_size = word_embedding_size + title_embedding_size + genre_embedding_size
    sg_hidden_size = 512
    sg_output_size = vocabulary_size
    sentence_generator = SentenceGenerator(sg_input_size, sg_hidden_size, sg_output_size)
    sentence_generator = cudalize(sentence_generator)
    sentence_generator.train()

    batch_size = 10
    learning_rate = 0.001
    num_epoch = 500
    print_every = 5

    trainEpochs(sentence_encoder, lyric_encoder, lyric_generator, sentence_generator, batch_size, learning_rate, num_epoch, print_every)
