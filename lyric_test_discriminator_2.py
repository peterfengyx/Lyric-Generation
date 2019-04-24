import pdb
from lyric_models import *
import pickle
import numpy as np
# from gensim import corpora

import torch.utils.data as data_utils
from torch.autograd import grad
import sys
import os

# input from command line
if len(sys.argv) != 2:
    raise ValueError("Wrong argument number!")

BatchSize = int(sys.argv[1]) # 20
LearningRate = 0.0001

print ('BatchSize: ', BatchSize)

# --------------------------- Load Data ---------------------------
train_set = pickle.load(open('data_new/training_012','rb'))
test_set = pickle.load(open('data_new/test_012','rb'))

test_idx_150_1 = pickle.load(open('data_new/test_idx_150_1.pkl','rb'))
test_idx_150_2 = pickle.load(open('data_new/test_idx_150_2.pkl','rb'))
test_idx_150_3 = pickle.load(open('data_new/test_idx_150_3.pkl','rb'))

# pdb.set_trace()
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
# title size
TitleSize = 300
# the number of iterations of the discriminator per generator iteration
NumDisIter = 1
#----------------------------------------------------------------
# load dictionary
# idx2word = corpora.Dictionary.load('data_new/dict.txt')
# load w2v vectors
# idx2vec = pickle.load(open('data_new/w2v.pkl','rb'))
word_embedding = np.eye(DictionarySize)
title_embedding = pickle.load(open('data_new/w2v_embedding.pkl','rb'))
genre_embedding = torch.eye(GenreSize)
line_end_embedding = torch.eye(MaxLineNum).type(torch.LongTensor)
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
              title_tensor,
              genre_tensor,
              real_lyric_tensor,
              real_line_length_tensor,
              real_line_num_tensor,
              sentence_encoder,
              lyric_encoder,
              lyric_generator,
              sentence_generator,
              lyric_discriminator,
              sentence_encoder_optimizer,
              lyric_encoder_optimizer,
              lyric_generator_optimizer,
              sentence_generator_optimizer,
              lyric_discriminator_optimizer,
              batch_size,
              max_line_number = MaxLineNum,
              max_line_length = MaxLineLen,
              num_discriminator_iter = NumDisIter):
              #  ,
              # lg_end_loss_weight = LgEndLossWeight,
              # sg_word_loss_weight = SgWordLossWeight):
    
    if model_type == 'train':
        sentence_encoder_optimizer.zero_grad()
        lyric_encoder_optimizer.zero_grad()
        lyric_generator_optimizer.zero_grad()
        sentence_generator_optimizer.zero_grad()
        lyric_discriminator_optimizer.zero_grad()
    
    # gan_loss_data = 0.0
    # generator_loss_data = 0.0
    discriminator_loss_data = 0.0

    # real lyric embedding
    real_line_number = torch.max(real_line_num_tensor).item()
    real_line_length = torch.max(real_line_length_tensor).item()

    real_le_hidden = cudalize(Variable(lyric_encoder.initHidden(batch_size))) # torch.Size([1, 10, 512])
    real_le_hiddens_variable = real_le_hidden # torch.Size([1, 10, 512])

    genre_embedding_tensor = genre_embedding[genre_tensor]
    
    for real_line_num in range(real_line_number):
        real_se_hidden = cudalize(Variable(sentence_encoder.initHidden(batch_size))) # torch.Size([1, 10, 512])
        real_se_hiddens_variable = real_se_hidden # torch.Size([1, 10, 512])
        
        for real_line_idx in range(real_line_length):
            real_se_word_tensor = torch.from_numpy(word_embedding[real_lyric_tensor[:,real_line_num,real_line_idx]]).type(torch.FloatTensor) # torch.Size([10, 9746])
            # title_tensor - this line, torch.Size([10, 9746])
            # genre_embedding_tensor - this line, torch.Size([10, 3])
            real_se_input = torch.cat((real_se_word_tensor, title_tensor, genre_embedding_tensor), 1) # torch.Size([10, 19495])
            real_se_input = cudalize(Variable(real_se_input))
            _, real_se_hidden = sentence_encoder(real_se_input, real_se_hidden, batch_size)
            real_se_hiddens_variable = torch.cat((real_se_hiddens_variable, real_se_hidden))

        real_line_latent_variable = real_se_hiddens_variable[real_line_length_tensor[:,real_line_num], np.arange(batch_size), :] # torch.Size([10, 512])
        real_le_title_tensor_variable = cudalize(Variable(title_tensor)) # torch.Size([10, 9746])
        real_le_genre_variable = cudalize(Variable(genre_embedding_tensor)) # torch.Size([10, 3])
        real_le_input = torch.cat((real_line_latent_variable, real_le_title_tensor_variable, real_le_genre_variable), 1) # torch.Size([10, 10261])

        _, real_le_hidden = lyric_encoder(real_le_input, real_le_hidden, batch_size)
        real_le_hiddens_variable = torch.cat((real_le_hiddens_variable, real_le_hidden))
    
    # real_lyric_latent_variable
    real_lyric_latent_variable = real_le_hiddens_variable[real_line_num_tensor, np.arange(batch_size), :] # torch.Size([10, 512])

    # generated lyric embedding
    noise_un_variable = cudalize(Variable(torch.randn(real_lyric_latent_variable.size()))) # torch.Size([10, 512])
    # normalize
    noise_mean_variable = torch.mean(noise_un_variable, dim=1, keepdim=True)
    noise_std_variable = torch.std(noise_un_variable, dim=1, keepdim=True)
    noise_variable = (noise_un_variable - noise_mean_variable)/noise_std_variable

    lg_temp_variable = noise_variable

    softmax = nn.Softmax(dim=1)

    lg_hidden = cudalize(Variable(lyric_generator.initHidden(batch_size))) # torch.Size([1, 10, 512])

    lg_outputs_length = np.array([max_line_number]*batch_size) # (10,)
    lg_length_flag = np.ones(batch_size, dtype=int) # (10,)

    le_hidden = cudalize(Variable(lyric_encoder.initHidden(batch_size))) # torch.Size([1, 10, 512])
    le_hiddens_variable = le_hidden # torch.Size([1, 10, 512])

    for line_num in range(max_line_number):
        lg_title_tensor_variable = cudalize(Variable(title_tensor)) # torch.Size([10, 9746])
        lg_genre_variable = cudalize(Variable(genre_embedding_tensor)) # torch.Size([10, 3])
        lg_input = torch.cat((lg_temp_variable, lg_title_tensor_variable, lg_genre_variable), 1) # torch.Size([10, 10261])
        # lg_input = torch.cat((noise_variable, lg_title_tensor_variable, lg_genre_variable), 1) # torch.Size([10, 10261])

        end_output, topic_output, lg_hidden = lyric_generator(lg_input, lg_hidden, batch_size)

        # workable, but be careful! Now p = 0.5, need to change for other p-s!
        end_output_softmax = softmax(end_output)
        end_ni = np.argmax(end_output_softmax.data.cpu().numpy(), axis=1)

        end_batch_index = np.where(end_ni == 1)[0]
        if np.sum(lg_length_flag[end_batch_index]) > 0:
            lg_outputs_length[end_batch_index] = line_num + 1 # line_num starts from 0!
            lg_length_flag[end_batch_index] = 0
        
        sg_hidden = topic_output.view(1, batch_size, -1) # torch.Size([1, 10, 512])
        sg_hiddens_variable = sg_hidden

        sg_word_tensor = torch.from_numpy(np.array([word_embedding[SOS]]*batch_size)).type(torch.FloatTensor) # torch.Size([10, 9746])
        # sg_word_outputs = cudalize(Variable(torch.zeros(line_length-1, batch_size, sentence_generator.output_size))) # torch.Size([19, 10, 9746])
        se_hidden = cudalize(Variable(sentence_encoder.initHidden(batch_size))) # torch.Size([1, 10, 512])
        # genre_embedding_tensor # torch.Size([10, 3])
        se_input = torch.cat((softmax(sg_word_tensor), title_tensor, genre_embedding_tensor), 1) # torch.Size([10, 19495])
        se_input = cudalize(Variable(se_input))
        _, se_hidden = sentence_encoder(se_input, se_hidden, batch_size)
        se_hiddens_variable = se_hidden # torch.Size([1, 10, 512])

        sg_outputs_length = np.array([max_line_length-1]*batch_size)
        sg_length_flag = np.ones(batch_size, dtype=int)

        for line_idx in range(1, max_line_length):
            # title_tensor - this line
            # genre_embedding_tensor - this line, torch.Size([10, 3])
            sg_input = torch.cat((sg_word_tensor, title_tensor, genre_embedding_tensor), 1) # torch.Size([10, 19495])
            sg_input = cudalize(Variable(sg_input))

            sg_output, sg_hidden = sentence_generator(sg_input, sg_hidden, batch_size)
            sg_hiddens_variable = torch.cat((sg_hiddens_variable, sg_hidden))

            sg_output_softmax = softmax(sg_output)

            ni = torch.multinomial(sg_output_softmax, 1).cpu().view(-1)
            # _, topi = sg_output_softmax.topk(1)
            # ni = topi.cpu().view(-1) # workable, but be careful
            sg_word_tensor = torch.from_numpy(word_embedding[ni]).type(torch.FloatTensor)
            
            eos_ni = ni.numpy()
            # be careful about <SOS>!!!!!!
            eos_batch_index = np.where(eos_ni == EOS)[0]
            if np.sum(sg_length_flag[eos_batch_index]) > 0:
                sg_outputs_length[eos_batch_index] = line_idx # exclude <SOS>, but include <EOS>
                sg_length_flag[eos_batch_index] = 0

            se_title_variable = cudalize(Variable(title_tensor))
            se_genre_variable = cudalize(Variable(genre_embedding_tensor))
            
            se_input = torch.cat((sg_output_softmax, se_title_variable, se_genre_variable), 1) # torch.Size([10, 19495])
            _, se_hidden = sentence_encoder(se_input, se_hidden, batch_size)
            se_hiddens_variable = torch.cat((se_hiddens_variable, se_hidden))
        
        lg_temp_variable = sg_hiddens_variable[sg_outputs_length, np.arange(batch_size), :]
        
        line_latent_variable = se_hiddens_variable[sg_outputs_length, np.arange(batch_size), :] # torch.Size([10, 512])
        le_title_tensor_variable = cudalize(Variable(title_tensor)) # torch.Size([10, 9746])
        le_genre_variable = cudalize(Variable(genre_embedding_tensor)) # torch.Size([10, 3])
        le_input = torch.cat((line_latent_variable, le_title_tensor_variable, le_genre_variable), 1) # torch.Size([10, 10261])

        _, le_hidden = lyric_encoder(le_input, le_hidden, batch_size)
        le_hiddens_variable = torch.cat((le_hiddens_variable, le_hidden))

    # generated_lyric_latent_variable
    generated_lyric_latent_variable = le_hiddens_variable[lg_outputs_length, np.arange(batch_size), :] # torch.Size([10, 512])
    
    # Now the two variables prepared, dig into gan training procedure.
    # GAN starts
    D_result_real = lyric_discriminator(real_lyric_latent_variable).squeeze() # does the .squeeze() really needed?
    D_real_loss = -torch.mean(D_result_real)

    D_result_fake = lyric_discriminator(generated_lyric_latent_variable).squeeze() # does the .squeeze() really needed?
    D_fake_loss = torch.mean(D_result_fake)

    # D_result_fake = lyric_discriminator(real_lyric_latent_variable).squeeze() # does the .squeeze() really needed?
    # D_fake_loss = torch.mean(D_result_fake)

    discriminator_loss = D_real_loss + D_fake_loss
    discriminator_loss_data = discriminator_loss.item()

    return D_real_loss.item(), D_fake_loss.item(), -discriminator_loss_data

def trainEpochs(sentence_encoder, 
                lyric_encoder, 
                lyric_generator, 
                sentence_generator, 
                lyric_discriminator, 
                batch_size, 
                learning_rate, 
                num_epoch, 
                print_every):
    sentence_encoder_optimizer = torch.optim.Adam(sentence_encoder.parameters(), lr=learning_rate*0.95)
    lyric_encoder_optimizer = torch.optim.Adam(lyric_encoder.parameters(), lr=learning_rate*0.95)
    lyric_generator_optimizer = torch.optim.Adam(lyric_generator.parameters(), lr=learning_rate*0.95)
    sentence_generator_optimizer = torch.optim.Adam(sentence_generator.parameters(), lr=learning_rate*0.95)
    lyric_discriminator_optimizer = torch.optim.Adam(lyric_discriminator.parameters(), lr=learning_rate)

    val_loader = data_utils.DataLoader(dataset=LyricDataset(test_set[test_idx_150_2]),
                                       batch_size=batch_size,
                                       shuffle=True)    
    sentence_encoder.eval()
    lyric_encoder.eval()
    lyric_generator.eval()
    sentence_generator.eval()
    lyric_discriminator.eval()    
    
    validation_loss_gan_list = []
    validation_loss_gene_list = []
    validation_loss_disc_list = []

    for val_batch, val_data in enumerate(val_loader, 0):
        title_tensor = val_data['title'].type(torch.FloatTensor) # torch.Size([10, 9746])
        genre_tensor = val_data['genre'] # torch.Size([10]), tensor([0, 2, 1, 1, 0, 1, 2, 1, 1, 1])
        lyric_tensor = val_data['lyric'] # torch.Size([10, 40, 32])
        line_length_tensor = val_data['line_length'] # torch.Size([10, 40])
        line_num_tensor = val_data['line_numb'] # torch.Size([10]), tensor([40, 17, 31, 38, 40, 40, 22,  9, 12, 39])

        print (val_batch)
        gan_loss, gene_loss, disc_loss = train_val('val',
                                                    title_tensor,
                                                    genre_tensor,
                                                    lyric_tensor,
                                                    line_length_tensor,
                                                    line_num_tensor,
                                                    sentence_encoder,
                                                    lyric_encoder,
                                                    lyric_generator,
                                                    sentence_generator,
                                                    lyric_discriminator,
                                                    sentence_encoder_optimizer,
                                                    lyric_encoder_optimizer,
                                                    lyric_generator_optimizer,
                                                    sentence_generator_optimizer,
                                                    lyric_discriminator_optimizer,
                                                    len(line_num_tensor))
            
        validation_loss_gan_list.append(gan_loss)
        validation_loss_gene_list.append(gene_loss)
        validation_loss_disc_list.append(disc_loss)
        
    print_loss_gan_avg_val = np.mean(np.array(validation_loss_gan_list))
    print_loss_gene_avg_val = np.mean(np.array(validation_loss_gene_list))
    print_loss_disc_avg_val = np.mean(np.array(validation_loss_disc_list))

    print('        Validation loss: [%.6f, %.6f, %.6f]' % (print_loss_gan_avg_val, print_loss_gene_avg_val, print_loss_disc_avg_val))

if __name__=='__main__':
    word_embedding_size = DictionarySize
    title_embedding_size = TitleSize
    genre_embedding_size = GenreSize

    saving_dir_gen = "lyric_gan_25"
    epoch_num_gen = 4

    # saving_dir_gen = "lyric_gan_scratch_128_25"
    # epoch_num_gen = 22

    # saving_dir_gen = "tf_autoencoder_128_30_01"
    # epoch_num_gen = 20

    saving_dir_dis = "lyric_gan_25"
    epoch_num_dis = 4

    lyric_latent_size = 128
    # lyric generator - lg
    lg_input_size = lyric_latent_size + title_embedding_size + genre_embedding_size
    lg_embedding_size = 128 # not used
    lg_hidden_size = 128 # 512
    lg_topic_latent_size = 128 # 512
    lg_topic_output_size = 128 # 512
    lyric_generator = LyricGenerator(lg_input_size, lg_embedding_size, lg_hidden_size, lg_topic_latent_size, lg_topic_output_size)
    # need to load the weights
    # lyric_generator.load_state_dict(torch.load(saving_dir_gen+'/lyric_generator_'+str(epoch_num_gen)))
    lyric_generator = cudalize(lyric_generator)
    lyric_generator.eval()

    # sentence generator - sg
    sg_input_size = word_embedding_size + title_embedding_size + genre_embedding_size
    sg_embedding_size = 128
    sg_hidden_size = lg_topic_output_size # 512
    sg_output_size = DictionarySize
    sentence_generator = SentenceGenerator(sg_input_size, sg_embedding_size, sg_hidden_size, sg_output_size)
    # need to load the weights
    # sentence_generator.load_state_dict(torch.load(saving_dir_gen+'/sentence_generator_'+str(epoch_num_gen)))
    sentence_generator = cudalize(sentence_generator)
    sentence_generator.eval()

    # sentence encoder - se
    se_input_size = word_embedding_size + title_embedding_size + genre_embedding_size
    se_embedding_size = 128
    se_hidden_size = 128 # 512
    sentence_encoder = SentenceEncoder(se_input_size, se_embedding_size, se_hidden_size)
    # need to load the weights
    sentence_encoder.load_state_dict(torch.load(saving_dir_dis+'/sentence_encoder_'+str(epoch_num_dis)))
    sentence_encoder = cudalize(sentence_encoder)
    sentence_encoder.eval()

    # lyric encoder - le
    le_input_size = se_hidden_size + title_embedding_size + genre_embedding_size
    le_embedding_size = 128 # not used
    le_hidden_size = lyric_latent_size
    lyric_encoder = LyricEncoder(le_input_size, le_embedding_size, le_hidden_size)
    # need to load the weights
    lyric_encoder.load_state_dict(torch.load(saving_dir_dis+'/lyric_encoder_'+str(epoch_num_dis)))
    lyric_encoder = cudalize(lyric_encoder)
    lyric_encoder.eval()

    # lyric discriminator - ldis
    ldis_input_size = lyric_latent_size
    lyric_discriminator = LyricDiscriminator(ldis_input_size)
    # need to load the weights
    lyric_discriminator.load_state_dict(torch.load(saving_dir_dis+'/lyric_discriminator_'+str(epoch_num_dis)))
    lyric_discriminator = cudalize(lyric_discriminator)
    lyric_discriminator.eval()

    batch_size = BatchSize # 20
    learning_rate = LearningRate
    num_epoch = 1000
    print_every = 1
    
    trainEpochs(sentence_encoder, lyric_encoder, lyric_generator, sentence_generator, lyric_discriminator, batch_size, learning_rate, num_epoch, print_every)
