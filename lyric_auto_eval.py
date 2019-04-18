import pdb

from lyric_models import *
import pickle
import numpy as np
import torch.utils.data as data_utils
from gensim import corpora
# --------------------------- Load Data ---------------------------
train_set = pickle.load(open('data_new/training_012','rb'))
val_set = pickle.load(open('data_new/valid_012','rb'))
test_set = pickle.load(open('data_new/test_012','rb'))
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
word_embedding = np.eye(DictionarySize)
title_embedding = pickle.load(open('data_new/w2v_embedding.pkl','rb'))
genre_embedding = torch.eye(GenreSize)
line_end_embedding = torch.eye(MaxLineNum).type(torch.LongTensor)
idx2word = corpora.Dictionary.load('data_new/dict.txt')
#########################################################################

class LyricDataset(data_utils.Dataset):
    def __init__(self, lyric_set, max_line_num = MaxLineNum):
        self.lyric_set = lyric_set
        self.max_line_num = max_line_num
        self.len = len(lyric_set)
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        o_title = np.array(self.lyric_set[index][0])
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
        
        return {'o_title': o_title, 'title': title, 'genre': genre, 'lyric': np.array(lyric), 'line_length': np.array(line_length), 'line_numb': line_numb}

#########################################################################

def lyric_generate(title_tensor,
                   genre_tensor,
                   lyric_tensor,
                   line_length_tensor,
                   line_num_tensor,
                   sentence_encoder, # se
                   lyric_encoder, # le
                   lyric_generator, # lg
                   sentence_generator, # sg
                   batch_size,
                   max_line_number = MaxLineNum,
                   max_line_length = MaxLineLen):

    line_number = torch.max(line_num_tensor).item()
    line_length = torch.max(line_length_tensor).item()

    le_hidden = cudalize(Variable(lyric_encoder.initHidden(batch_size))) # torch.Size([1, 10, 512])
    le_hiddens_variable = le_hidden # torch.Size([1, 10, 512])

    genre_embedding_tensor = genre_embedding[genre_tensor]
    
    for line_num in range(line_number):
        se_hidden = cudalize(Variable(sentence_encoder.initHidden(batch_size))) # torch.Size([1, 10, 512])
        se_hiddens_variable = se_hidden # torch.Size([1, 10, 512])
        
        for line_idx in range(line_length):
            se_word_tensor = torch.from_numpy(word_embedding[lyric_tensor[:,line_num,line_idx]]).type(torch.FloatTensor) # torch.Size([10, 9746])
            # title_tensor - this line, torch.Size([10, 9746])
            # genre_embedding_tensor - this line, torch.Size([10, 3])
            # only support batch size = 1 !!!
            se_word_tensor = se_word_tensor.view(1, -1)

            se_input = torch.cat((se_word_tensor, title_tensor, genre_embedding_tensor), 1) # torch.Size([10, 19495])
            se_input = cudalize(Variable(se_input))
            _, se_hidden = sentence_encoder(se_input, se_hidden, batch_size)
            se_hiddens_variable = torch.cat((se_hiddens_variable, se_hidden))

        line_latent_variable = se_hiddens_variable[line_length_tensor[:,line_num], np.arange(batch_size), :] # torch.Size([10, 512])
        le_title_tensor_variable = cudalize(Variable(title_tensor)) # torch.Size([10, 9746])
        le_genre_variable = cudalize(Variable(genre_embedding_tensor)) # torch.Size([10, 3])
        le_input = torch.cat((line_latent_variable, le_title_tensor_variable, le_genre_variable), 1) # torch.Size([10, 10261])

        _, le_hidden = lyric_encoder(le_input, le_hidden, batch_size)
        le_hiddens_variable = torch.cat((le_hiddens_variable, le_hidden))
    
    lyric_un_latent_variable = le_hiddens_variable[line_num_tensor, np.arange(batch_size), :] # torch.Size([10, 512])
    # need to normalize lyric_un_latent_variable, 0 mean, std = 1
    lyric_mean_variable = torch.mean(lyric_un_latent_variable, dim=1, keepdim=True)
    lyric_std_variable = torch.std(lyric_un_latent_variable, dim=1, keepdim=True)
    lyric_latent_variable = (lyric_un_latent_variable - lyric_mean_variable)/lyric_std_variable
    # pdb.set_trace()

    # need to do decoder on lyric_latent_variable
    softmax = nn.Softmax(dim=1)

    lg_hidden = cudalize(Variable(lyric_generator.initHidden(batch_size))) # torch.Size([1, 10, 512])
    # lg_end_outputs = cudalize(Variable(torch.zeros(line_number, batch_size, 2))) #torch.Size([40, 10, 2])

    # only support batch_size = 1
    generated_lyric_list = []
    generated_line_num = 0

    for line_num in range(max_line_number):
        lg_title_tensor_variable = cudalize(Variable(title_tensor)) # torch.Size([10, 9746])
        lg_genre_variable = cudalize(Variable(genre_embedding_tensor)) # torch.Size([10, 3])
        lg_input = torch.cat((lyric_latent_variable, lg_title_tensor_variable, lg_genre_variable), 1) # torch.Size([10, 10261])

        end_output, topic_output, lg_hidden = lyric_generator(lg_input, lg_hidden, batch_size)
        # lg_end_outputs[line_num] = end_output

        end_output_softmax = softmax(end_output)
        # only works for  batch size = 1 !!!
        if end_output_softmax[0][1].item() < 0.5:
            generated_line_num += 1

        sg_hidden = topic_output.view(1, batch_size, -1) # torch.Size([1, 10, 512])
        sg_word_tensor = torch.from_numpy(np.array([word_embedding[SOS]]*batch_size)).type(torch.FloatTensor) # torch.Size([10, 9746])
        # sg_word_outputs = cudalize(Variable(torch.zeros(line_length-1, batch_size, sentence_generator.output_size))) # torch.Size([19, 10, 9746])

        generated_line_list = []
        for line_idx in range(1, max_line_length):
            # title_tensor - this line
            # genre_embedding_tensor - this line, torch.Size([10, 3])
            # pdb.set_trace()
            sg_input = torch.cat((sg_word_tensor, title_tensor, genre_embedding_tensor), 1) # torch.Size([10, 19495])
            sg_input = cudalize(Variable(sg_input))

            sg_output, sg_hidden = sentence_generator(sg_input, sg_hidden, batch_size)
            # sg_word_outputs[line_idx-1] = sg_output

            # only works for  batch size = 1 !!!
            prob = softmax(sg_output).view(-1).cpu().data.numpy()
            ni = np.random.choice(DictionarySize, 1, p=prob)

            # pdb.set_trace()
            # _, topi = softmax(sg_output).topk(1)
            # ni = topi.cpu().view(-1) # workable, but be careful

            sg_word_tensor = torch.from_numpy(word_embedding[ni]).type(torch.FloatTensor)
            # only works for  batch size = 1 !!!
            sg_word_tensor = sg_word_tensor.view(1, -1)
            generated_line_list.append(ni.item())
        generated_lyric_list.append(generated_line_list)
    
    print ("----Generated Lyric--------------------------------------")
    print ('Generated Line Number: ', generated_line_num)
    print_lyric(generated_lyric_list)

#########################################################################

# only works for  batch size = 1 !!!
def print_one_line(idx_list):
    word_list = [idx2word[idx] for idx in idx_list]
    print (word_list)

def print_title(title_list):
    print_one_line(title_list)

def print_genre(genre_num):
    genre_list = ['Hip-Hop:0', 'Metal:1', 'Country:2']
    print (genre_list[genre_num])

def print_lyric(lyric_list):
    for entry in lyric_list:
        print_one_line(entry) 

#########################################################################

def trainGenerate(d_set, sentence_encoder, lyric_encoder, lyric_generator, sentence_generator, batch_size):
    # generate for training data
    d_loader = data_utils.DataLoader(dataset=LyricDataset(d_set), batch_size=batch_size, shuffle=True)
    sentence_encoder.eval()
    lyric_encoder.eval()
    lyric_generator.eval()
    sentence_generator.eval()
        
    for _, data in enumerate(d_loader, 0):
        title_tensor = data['title'].type(torch.FloatTensor) # torch.Size([10, 9746])
        genre_tensor = data['genre'] # torch.Size([10]), tensor([0, 2, 1, 1, 0, 1, 2, 1, 1, 1])
        lyric_tensor = data['lyric'] # torch.Size([10, 40, 32])
        line_length_tensor = data['line_length'] # torch.Size([10, 40])
        line_num_tensor = data['line_numb'] # torch.Size([10]), tensor([40, 17, 31, 38, 40, 40, 22,  9, 12, 39])

        # only works for  batch size = 1 !!!
        o_title_list = data['o_title'][0].tolist()
        print_title(o_title_list)
        print_genre(genre_tensor.item())
        print ("----Real Lyric--------------------------------------")
        print ('Real Line Number: ', line_num_tensor.item())
        print_lyric(lyric_tensor[0][:line_num_tensor].tolist())

        lyric_generate(title_tensor,
                       genre_tensor,
                       lyric_tensor,
                       line_length_tensor,
                       line_num_tensor,
                       sentence_encoder, # se
                       lyric_encoder, # le
                       lyric_generator, # lg
                       sentence_generator, # sg
                       len(line_num_tensor))
        
        pdb.set_trace()

#########################################################################
'''
# for the 128 model
word_embedding_size = DictionarySize
title_embedding_size = TitleSize
genre_embedding_size = GenreSize

d_set = val_set
saving_dir = "tf_autoencoder_128_30_01"
epoch_num = 9

batch_size = 1
# sentence encoder - se
se_input_size = word_embedding_size + title_embedding_size + genre_embedding_size
se_embedding_size = 128
se_hidden_size = 128 # 512
sentence_encoder = SentenceEncoder(se_input_size, se_embedding_size, se_hidden_size)
sentence_encoder.load_state_dict(torch.load(saving_dir+'/sentence_encoder_'+str(epoch_num)))
sentence_encoder = cudalize(sentence_encoder)
sentence_encoder.eval()

# lyric encoder - le
le_input_size = se_hidden_size + title_embedding_size + genre_embedding_size
le_embedding_size = 128 # not used
le_hidden_size = 128
lyric_encoder = LyricEncoder(le_input_size, le_embedding_size, le_hidden_size)
lyric_encoder.load_state_dict(torch.load(saving_dir+'/lyric_encoder_'+str(epoch_num)))
lyric_encoder = cudalize(lyric_encoder)
lyric_encoder.eval()

# lyric generator - lg
lg_input_size = le_hidden_size + title_embedding_size + genre_embedding_size
lg_embedding_size = 128 # not used
lg_hidden_size = 128 # 512
lg_topic_latent_size = 128 # 512
lg_topic_output_size = 128 # 512
lyric_generator = LyricGenerator(lg_input_size, lg_embedding_size, lg_hidden_size, lg_topic_latent_size, lg_topic_output_size)
lyric_generator.load_state_dict(torch.load(saving_dir+'/lyric_generator_'+str(epoch_num)))
lyric_generator = cudalize(lyric_generator)
lyric_generator.eval()

# sentence generator - sg
sg_input_size = word_embedding_size + title_embedding_size + genre_embedding_size
sg_embedding_size = 128
sg_hidden_size = lg_topic_output_size # 512
sg_output_size = DictionarySize
sentence_generator = SentenceGenerator(sg_input_size, sg_embedding_size, sg_hidden_size, sg_output_size)
sentence_generator.load_state_dict(torch.load(saving_dir+'/sentence_generator_'+str(epoch_num)))
sentence_generator = cudalize(sentence_generator)
sentence_generator.eval()

trainGenerate(d_set, sentence_encoder, lyric_encoder, lyric_generator, sentence_generator, batch_size)
'''
# for the 256 model
word_embedding_size = DictionarySize
title_embedding_size = TitleSize
genre_embedding_size = GenreSize

d_set = val_set
saving_dir = "tf_autoencoder_256_25_1"
epoch_num = 14

batch_size = 1
# sentence encoder - se
se_input_size = word_embedding_size + title_embedding_size + genre_embedding_size
se_embedding_size = 128
se_hidden_size = 256 # 512
sentence_encoder = SentenceEncoder(se_input_size, se_embedding_size, se_hidden_size)
sentence_encoder.load_state_dict(torch.load(saving_dir+'/sentence_encoder_'+str(epoch_num)))
sentence_encoder = cudalize(sentence_encoder)
sentence_encoder.eval()

# lyric encoder - le
le_input_size = se_hidden_size + title_embedding_size + genre_embedding_size
le_embedding_size = 256 # not used
le_hidden_size = 256
lyric_encoder = LyricEncoder(le_input_size, le_embedding_size, le_hidden_size)
lyric_encoder.load_state_dict(torch.load(saving_dir+'/lyric_encoder_'+str(epoch_num)))
lyric_encoder = cudalize(lyric_encoder)
lyric_encoder.eval()

# lyric generator - lg
lg_input_size = le_hidden_size + title_embedding_size + genre_embedding_size
lg_embedding_size = 256 # not used
lg_hidden_size = 256 # 512
lg_topic_latent_size = 256 # 512
lg_topic_output_size = 256 # 512
lyric_generator = LyricGenerator(lg_input_size, lg_embedding_size, lg_hidden_size, lg_topic_latent_size, lg_topic_output_size)
lyric_generator.load_state_dict(torch.load(saving_dir+'/lyric_generator_'+str(epoch_num)))
lyric_generator = cudalize(lyric_generator)
lyric_generator.eval()

# sentence generator - sg
sg_input_size = word_embedding_size + title_embedding_size + genre_embedding_size
sg_embedding_size = 128
sg_hidden_size = lg_topic_output_size # 512
sg_output_size = DictionarySize
sentence_generator = SentenceGenerator(sg_input_size, sg_embedding_size, sg_hidden_size, sg_output_size)
sentence_generator.load_state_dict(torch.load(saving_dir+'/sentence_generator_'+str(epoch_num)))
sentence_generator = cudalize(sentence_generator)
sentence_generator.eval()

trainGenerate(d_set, sentence_encoder, lyric_encoder, lyric_generator, sentence_generator, batch_size)
