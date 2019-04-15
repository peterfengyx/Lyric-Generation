import pdb
from lyric_models import *
import pickle
import numpy as np
from gensim import corpora
from tensorboardX import SummaryWriter

import torch.utils.data as data_utils
from torch.autograd import grad

# input from command line
# LgEndLossWeight = 5 # 10
# SgWordLossWeight = 1
SavingDir = "."
LearningRate = 0.0001
# --------------------------- Load Data ---------------------------
train_set = pickle.load(open('data/training_012','rb'))
val_set = pickle.load(open('data/valid_012','rb'))
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
#----------------------------------------------------------------
# load dictionary
# idx2word = corpora.Dictionary.load('data/dict.txt')
# load w2v vectors
# idx2vec = pickle.load(open('data/w2v.pkl','rb'))
word_embedding = np.eye(DictionarySize)
title_embedding = pickle.load(open('data/w2v_embedding.pkl','rb'))
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
              max_line_length = MaxLineLen):
              #  ,
              # lg_end_loss_weight = LgEndLossWeight,
              # sg_word_loss_weight = SgWordLossWeight):
    
    if model_type == 'train':
        sentence_encoder_optimizer.zero_grad()
        lyric_encoder_optimizer.zero_grad()
        lyric_generator_optimizer.zero_grad()
        sentence_generator_optimizer.zero_grad()
        lyric_discriminator_optimizer.zero_grad()
    
    gan_loss_data = 0.0
    generator_loss_data = 0.0
    discriminator_loss_data = 0.0

    # real lyric embedding
    real_line_number = torch.max(real_line_num_tensor).item()
    real_line_length = torch.max(real_line_length_tensor).item()

    real_le_hidden = cudalize(Variable(lyric_encoder.initHidden(batch_size))) # torch.Size([1, 10, 512])
    real_le_hiddens_variable = real_le_hidden # torch.Size([1, 10, 512])
    
    for real_line_num in range(real_line_number):
        real_se_hidden = cudalize(Variable(sentence_encoder.initHidden(batch_size))) # torch.Size([1, 10, 512])
        real_se_hiddens_variable = real_se_hidden # torch.Size([1, 10, 512])
        
        for real_line_idx in range(real_line_length):
            real_se_word_tensor = torch.from_numpy(word_embedding[real_lyric_tensor[:,real_line_num,real_line_idx]]).type(torch.FloatTensor) # torch.Size([10, 9746])
            # title_tensor - this line, torch.Size([10, 9746])
            real_se_genre_tensor = genre_embedding[genre_tensor] # torch.Size([10, 3])
            real_se_input = torch.cat((real_se_word_tensor, title_tensor, real_se_genre_tensor), 1) # torch.Size([10, 19495])
            real_se_input = cudalize(Variable(real_se_input))
            _, real_se_hidden = sentence_encoder(real_se_input, real_se_hidden, batch_size)
            real_se_hiddens_variable = torch.cat((real_se_hiddens_variable, real_se_hidden))

        real_line_latent_variable = real_se_hiddens_variable[real_line_length_tensor[:,real_line_num], np.arange(batch_size), :] # torch.Size([10, 512])
        real_le_title_tensor_variable = cudalize(Variable(title_tensor)) # torch.Size([10, 9746])
        real_le_genre_variable = cudalize(Variable(genre_embedding[genre_tensor])) # torch.Size([10, 3])
        real_le_input = torch.cat((real_line_latent_variable, real_le_title_tensor_variable, real_le_genre_variable), 1) # torch.Size([10, 10261])

        _, real_le_hidden = lyric_encoder(real_le_input, real_le_hidden, batch_size)
        real_le_hiddens_variable = torch.cat((real_le_hiddens_variable, real_le_hidden))
    
    # real_lyric_latent_variable
    real_lyric_latent_variable = real_le_hiddens_variable[real_line_num_tensor, np.arange(batch_size), :] # torch.Size([10, 512])

    # generated lyric embedding
    # this line needs to be careful!!!
    noise_variable = cudalize(Variable(torch.randn(real_lyric_latent_variable.size()))) # torch.Size([10, 512])
    softmax = nn.Softmax(dim=1)

    lg_hidden = cudalize(Variable(lyric_generator.initHidden(batch_size))) # torch.Size([1, 10, 512])

    lg_outputs_length = np.array([max_line_number]*batch_size) # (10,)
    lg_length_flag = np.ones(batch_size, dtype=int) # (10,)

    le_hidden = cudalize(Variable(lyric_encoder.initHidden(batch_size))) # torch.Size([1, 10, 512])
    le_hiddens_variable = le_hidden # torch.Size([1, 10, 512])

    for line_num in range(max_line_number):
        lg_title_tensor_variable = cudalize(Variable(title_tensor)) # torch.Size([10, 9746])
        lg_genre_variable = cudalize(Variable(genre_embedding[genre_tensor])) # torch.Size([10, 3])
        lg_input = torch.cat((noise_variable, lg_title_tensor_variable, lg_genre_variable), 1) # torch.Size([10, 10261])

        end_output, topic_output, lg_hidden = lyric_generator(lg_input, lg_hidden, batch_size)
        # lg_end_outputs[line_num] = end_output

        # workable, but be careful! Now p = 0.5, need to change for other p-s!
        end_output_softmax = softmax(end_output)
        end_ni = np.argmax(end_output_softmax.data.cpu().numpy(), axis=1)

        end_batch_index = np.where(end_ni == 1)[0]
        if np.sum(lg_length_flag[end_batch_index]) > 0:
            lg_outputs_length[end_batch_index] = line_num
            lg_length_flag[end_batch_index] = 0

        sg_hidden = topic_output.view(1, batch_size, -1) # torch.Size([1, 10, 512])
        sg_word_tensor = torch.from_numpy(np.array([word_embedding[SOS]]*batch_size)).type(torch.FloatTensor) # torch.Size([10, 9746])
        # sg_word_outputs = cudalize(Variable(torch.zeros(line_length-1, batch_size, sentence_generator.output_size))) # torch.Size([19, 10, 9746])
        sg_outputs_length = np.array([max_line_length-1]*batch_size)
        sg_length_flag = np.ones(batch_size, dtype=int)

        se_hidden = cudalize(Variable(sentence_encoder.initHidden(batch_size))) # torch.Size([1, 10, 512])
        se_hiddens_variable = se_hidden # torch.Size([1, 10, 512])

        # check till here
        pdb.set_trace()

        for line_idx in range(1, max_line_length):
            # title_tensor - this line
            sg_genre_tensor = genre_embedding[genre_tensor] # torch.Size([10, 3])
            sg_input = torch.cat((sg_word_tensor, title_tensor, sg_genre_tensor), 1) # torch.Size([10, 19495])
            sg_input = cudalize(Variable(sg_input))

            sg_output, sg_hidden = sentence_generator(sg_input, sg_hidden, batch_size)
            # sg_word_outputs[line_idx-1] = sg_output

            sg_output_softmax = softmax(sg_output)
            _, topi = sg_output_softmax.topk(1)
            ni = topi.cpu().view(-1) # workable, but be careful
            sg_word_tensor = torch.from_numpy(word_embedding[ni]).type(torch.FloatTensor)

            # be careful, potential bug!
            eos_ni = np.argmax(sg_output_softmax.data.cpu().numpy(), axis=1) 
            pdb.set_trace()
            # be careful about <SOS>!!!!!!

            eos_batch_index = np.where(eos_ni == EOS)[0]
            if np.sum(sg_length_flag[eos_batch_index]) > 0:
                sg_outputs_length[eos_batch_index] = line_idx # exclude <SOS>, but include <EOS>
                sg_length_flag[eos_batch_index] = 0
            
            se_input = sg_output_softmax # Need to concat title and genre!!!
            _, se_hidden = sentence_encoder(se_input, se_hidden, batch_size)
            se_hiddens_variable = torch.cat((se_hiddens_variable, se_hidden))
        
        line_latent_variable = se_hiddens_variable[sg_outputs_length, np.arange(batch_size), :] # torch.Size([10, 512])
        le_title_tensor_variable = cudalize(Variable(title_tensor)) # torch.Size([10, 9746])
        le_genre_variable = cudalize(Variable(genre_embedding[genre_tensor])) # torch.Size([10, 3])
        le_input = torch.cat((line_latent_variable, le_title_tensor_variable, le_genre_variable), 1) # torch.Size([10, 10261])

        _, le_hidden = lyric_encoder(le_input, le_hidden, batch_size)
        le_hiddens_variable = torch.cat((le_hiddens_variable, le_hidden))

    # generated_lyric_latent_variable
    generated_lyric_latent_variable = le_hiddens_variable[lg_outputs_length, np.arange(batch_size), :] # torch.Size([10, 512])
    
    '''
    # GAN starts
    D_result_real = lyric_discriminator(real_lyric_latent_variable).squeeze()
    D_real_loss = -torch.mean(D_result_real)

    D_result_fake = lyric_discriminator(generated_lyric_latent_variable).squeeze()
    D_fake_loss = torch.mean(D_result_fake)

    # gradient penalty
    alpha = torch.rand((batch_size, 1, 1, 1))
    alpha = cudalize(alpha)

    x_hat = alpha * un_image_projected.data + (1 - alpha) * un_sentence_embedding.data
    x_hat.requires_grad = True

    pred_hat = lyric_discriminator(x_hat)

    gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=cudalize(torch.ones(pred_hat.size())), create_graph=True, retain_graph=True, only_inputs=True)[0]

    lambda_ = 10
    gradient_penalty = lambda_ * ((gradients.view(gradients.size()[0], -1).norm(2, 1) - 1) ** 2).mean()

    discriminator_loss = D_real_loss + D_fake_loss + gradient_penalty
    discriminator_loss_data = discriminator_loss.item()

    if model_type == "train":
        discriminator_loss.backward(retain_graph=True)
        lyric_discriminator_optimizer.step()

        projection_layer_optimizer.zero_grad()
        caption_decoder_optimizer.zero_grad()
        caption_encoder_optimizer.zero_grad()
        transform_layer_optimizer.zero_grad()
        distribution_discriminator_optimizer.zero_grad()
    
    D_result_generate = distribution_discriminator(un_sentence_embedding).squeeze()
    generator_loss = -torch.mean(D_result_generate)
    generator_loss_data = generator_loss.item()

    if model_type == 'train':
        generator_loss.backward(retain_graph=True)
    
    # GAN finishes

    if model_type == 'train':
        auto_loss.backward()
        sentence_encoder_optimizer.step()
        lyric_encoder_optimizer.step()
        lyric_generator_optimizer.step()
        sentence_generator_optimizer.step()
    '''
    
    # print(gan_loss_data, generator_loss_data, discriminator_loss_data)
    return gan_loss_data, generator_loss_data, discriminator_loss_data

def trainEpochs(sentence_encoder, 
                lyric_encoder, 
                lyric_generator, 
                sentence_generator, 
                lyric_discriminator, 
                batch_size, 
                learning_rate, 
                num_epoch, 
                print_every,
                saving_dir = SavingDir):
    sentence_encoder_optimizer = torch.optim.Adam(sentence_encoder.parameters(), lr=learning_rate)
    lyric_encoder_optimizer = torch.optim.Adam(lyric_encoder.parameters(), lr=learning_rate)
    lyric_generator_optimizer = torch.optim.Adam(lyric_generator.parameters(), lr=learning_rate)
    sentence_generator_optimizer = torch.optim.Adam(sentence_generator.parameters(), lr=learning_rate)
    lyric_discriminator_optimizer = torch.optim.Adam(lyric_discriminator.parameters(), lr=learning_rate)

    train_loader = data_utils.DataLoader(dataset=LyricDataset(train_set),
                                         batch_size=batch_size,
                                         shuffle=True) # True)
    val_loader = data_utils.DataLoader(dataset=LyricDataset(val_set),
                                       batch_size=batch_size,
                                       shuffle=True)
    iter_epoch = 0
    for epoch in range(num_epoch):
        sentence_encoder.train()
        lyric_encoder.train()
        lyric_generator.train()
        sentence_generator.train()
        lyric_discriminator.train()

        # print_loss_total_auto = 0.0  # Reset every print_every
        # print_loss_total_auto_list = []
        # print_loss_total_word = 0.0  # Reset every print_every
        # print_loss_total_word_list = []
        # print_loss_total_end = 0.0  # Reset every print_every
        # print_loss_total_end_list = []

        for batch, data in enumerate(train_loader, 0):
            title_tensor = data['title'].type(torch.FloatTensor) # torch.Size([10, 9746])
            genre_tensor = data['genre'] # torch.Size([10]), tensor([0, 2, 1, 1, 0, 1, 2, 1, 1, 1])
            lyric_tensor = data['lyric'] # torch.Size([10, 40, 32])
            line_length_tensor = data['line_length'] # torch.Size([10, 40])
            line_num_tensor = data['line_num'] # torch.Size([10]), tensor([40, 17, 31, 38, 40, 40, 22,  9, 12, 39])

            # print(batch)
            auto_loss, word_loss, end_loss = train_val('train',
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
        '''
        
            print_loss_total_auto += auto_loss
            print_loss_total_auto_list.append(auto_loss)
            print_loss_total_word += word_loss
            print_loss_total_word_list.append(word_loss)
            print_loss_total_end += end_loss
            print_loss_total_end_list.append(end_loss)

            if batch % print_every == (print_every-1):
                print_loss_avg_auto = print_loss_total_auto / print_every
                print_loss_total_auto = 0.0

                print_loss_avg_word = print_loss_total_word / print_every
                print_loss_total_word = 0.0

                print_loss_avg_end = print_loss_total_end / print_every
                print_loss_total_end = 0.0

                print('[%d, %d]  [%.6f, %.6f, %.6f]' % (epoch+1, batch+1, print_loss_avg_auto, print_loss_avg_word, print_loss_avg_end))
            
        print_loss_auto_avg_train = np.mean(np.array(print_loss_total_auto_list))
        print_loss_word_avg_train = np.mean(np.array(print_loss_total_word_list))
        print_loss_end_avg_train = np.mean(np.array(print_loss_total_end_list))

        print('Train loss: [%.6f, %.6f, %6f]' % (print_loss_auto_avg_train, print_loss_word_avg_train, print_loss_end_avg_train))
        
        
        # validation
        sentence_encoder.eval()
        lyric_encoder.eval()
        lyric_generator.eval()
        sentence_generator.eval()
        lyric_discriminator.eval()
        
        validation_loss_auto_list = []
        validation_loss_word_list = []
        validation_loss_end_list = []

        for _, val_data in enumerate(val_loader, 0):
            title_tensor = val_data['title'].type(torch.FloatTensor) # torch.Size([10, 9746])
            genre_tensor = val_data['genre'] # torch.Size([10]), tensor([0, 2, 1, 1, 0, 1, 2, 1, 1, 1])
            lyric_tensor = val_data['lyric'] # torch.Size([10, 40, 32])
            line_length_tensor = val_data['line_length'] # torch.Size([10, 40])
            line_num_tensor = val_data['line_num'] # torch.Size([10]), tensor([40, 17, 31, 38, 40, 40, 22,  9, 12, 39])

            auto_loss, word_loss, end_loss = train_val('val',
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
            validation_loss_auto_list.append(auto_loss)
            validation_loss_word_list.append(word_loss)
            validation_loss_end_list.append(end_loss)
        
        print_loss_auto_avg_val = np.mean(np.array(validation_loss_auto_list))
        print_loss_word_avg_val = np.mean(np.array(validation_loss_word_list))
        print_loss_end_avg_val = np.mean(np.array(validation_loss_end_list))

        print('        Validation loss: [%.6f, %.6f, %.6f]' % (print_loss_auto_avg_val, print_loss_word_avg_val, print_loss_end_avg_val))
        '''
        # write to tensorboard
        # iter_epoch += 1
        # writer.add_scalars(saving_dir+'/auto_loss/train_val_epoch', {'train': print_loss_auto_avg_train, 'val': print_loss_auto_avg_val}, iter_epoch)
        # writer.add_scalars(saving_dir+'/word_loss/train_val_epoch', {'train': print_loss_word_avg_train, 'val': print_loss_word_avg_val}, iter_epoch)
        # writer.add_scalars(saving_dir+'/end_loss/train_val_epoch', {'train': print_loss_end_avg_train, 'val': print_loss_end_avg_val}, iter_epoch)

        # save models    
        # torch.save(sentence_encoder.state_dict(), saving_dir+'/sentence_encoder_'+str(epoch+1))
        # torch.save(lyric_encoder.state_dict(), saving_dir+'/lyric_encoder_'+str(epoch+1))
        # torch.save(lyric_generator.state_dict(), saving_dir+'/lyric_generator_'+str(epoch+1))
        # torch.save(sentence_generator.state_dict(), saving_dir+'/sentence_generator_'+str(epoch+1))
        
if __name__=='__main__':
    word_embedding_size = DictionarySize
    title_embedding_size = TitleSize
    genre_embedding_size = GenreSize

    lyric_latent_size = 512
    # lyric generator - lg
    lg_input_size = lyric_latent_size + title_embedding_size + genre_embedding_size
    lg_embedding_size = 300
    lg_hidden_size = 300 # 512
    lg_topic_latent_size = 300 # 512
    lg_topic_output_size = 300 # 512
    lyric_generator = LyricGenerator(lg_input_size, lg_embedding_size, lg_hidden_size, lg_topic_latent_size, lg_topic_output_size)
    lyric_generator = cudalize(lyric_generator)
    # need to load the weights
    lyric_generator.train()

    # sentence generator - sg
    sg_input_size = word_embedding_size + title_embedding_size + genre_embedding_size
    sg_embedding_size = 300
    sg_hidden_size = 300 # 512
    sg_output_size = DictionarySize
    sentence_generator = SentenceGenerator(sg_input_size, sg_embedding_size, sg_hidden_size, sg_output_size)
    sentence_generator = cudalize(sentence_generator)
    # need to load the weights
    sentence_generator.train()

    # sentence encoder - se
    se_input_size = word_embedding_size + title_embedding_size + genre_embedding_size
    se_embedding_size = 300
    se_hidden_size = 300 # 512
    sentence_encoder = SentenceEncoder(se_input_size, se_embedding_size, se_hidden_size)
    sentence_encoder = cudalize(sentence_encoder)
    # need to load the weights
    sentence_encoder.train()

    # lyric encoder - le
    le_input_size = se_hidden_size + title_embedding_size + genre_embedding_size
    le_embedding_size = 300
    le_hidden_size = lyric_latent_size
    lyric_encoder = LyricEncoder(le_input_size, le_embedding_size, le_hidden_size)
    lyric_encoder = cudalize(lyric_encoder)
    # need to load the weights
    lyric_encoder.train()

    # lyric discriminator - ldis
    ldis_input_size = lyric_latent_size
    lyric_discriminator = LyricDiscriminator(ldis_input_size)
    lyric_discriminator = cudalize(lyric_discriminator)
    lyric_discriminator.train()

    batch_size = 10 # 20
    learning_rate = LearningRate
    num_epoch = 500
    print_every = 1
    
    # writer = SummaryWriter()
    trainEpochs(sentence_encoder, lyric_encoder, lyric_generator, sentence_generator, lyric_discriminator, batch_size, learning_rate, num_epoch, print_every)
    # writer.close()