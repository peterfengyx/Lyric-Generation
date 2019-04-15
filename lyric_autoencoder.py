import pdb
from lyric_models import *
import pickle
import numpy as np
from gensim import corpora
from tensorboardX import SummaryWriter

import torch.utils.data as data_utils

# input from command line
LgEndLossWeight = 10 # 10
SgWordLossWeight = 1
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
              lg_end_loss_weight = LgEndLossWeight,
              sg_word_loss_weight = SgWordLossWeight):
            #   ,
            #   max_line_num = MaxLineNum,
            #   max_line_length = MaxLineLen):
     
    if model_type == 'train':
        sentence_encoder_optimizer.zero_grad()
        lyric_encoder_optimizer.zero_grad()
        lyric_generator_optimizer.zero_grad()
        sentence_generator_optimizer.zero_grad()
    
    auto_loss_data = 0.0
    sg_word_loss_data = 0.0
    lg_end_loss_data = 0.0

    line_number = torch.max(line_num_tensor).item()
    line_length = torch.max(line_length_tensor).item()

    le_hidden = cudalize(Variable(lyric_encoder.initHidden(batch_size))) # torch.Size([1, 10, 512])
    le_hiddens_variable = le_hidden # torch.Size([1, 10, 512])
    
    for line_num in range(line_number):
        se_hidden = cudalize(Variable(sentence_encoder.initHidden(batch_size))) # torch.Size([1, 10, 512])
        se_hiddens_variable = se_hidden # torch.Size([1, 10, 512])
        
        for line_idx in range(line_length):
            se_word_tensor = torch.from_numpy(word_embedding[lyric_tensor[:,line_num,line_idx]]).type(torch.FloatTensor) # torch.Size([10, 9746])
            # title_tensor - this line, torch.Size([10, 9746])
            se_genre_tensor = genre_embedding[genre_tensor] # torch.Size([10, 3])
            se_input = torch.cat((se_word_tensor, title_tensor, se_genre_tensor), 1) # torch.Size([10, 19495])
            se_input = cudalize(Variable(se_input))
            _, se_hidden = sentence_encoder(se_input, se_hidden, batch_size)
            se_hiddens_variable = torch.cat((se_hiddens_variable, se_hidden))

        line_latent_variable = se_hiddens_variable[line_length_tensor[:,line_num], np.arange(batch_size), :] # torch.Size([10, 512])
        le_title_tensor_variable = cudalize(Variable(title_tensor)) # torch.Size([10, 9746])
        le_genre_variable = cudalize(Variable(genre_embedding[genre_tensor])) # torch.Size([10, 3])
        le_input = torch.cat((line_latent_variable, le_title_tensor_variable, le_genre_variable), 1) # torch.Size([10, 10261])

        _, le_hidden = lyric_encoder(le_input, le_hidden, batch_size)
        le_hiddens_variable = torch.cat((le_hiddens_variable, le_hidden))
    
    lyric_latent_variable = le_hiddens_variable[line_num_tensor, np.arange(batch_size), :] # torch.Size([10, 512])
    # need to do decoder on lyric_latent_variable
    softmax = nn.Softmax(dim=1)

    lg_hidden = cudalize(Variable(lyric_generator.initHidden(batch_size))) # torch.Size([1, 10, 512])
    lg_end_outputs = cudalize(Variable(torch.zeros(line_number, batch_size, 2))) #torch.Size([40, 10, 2])

    sg_word_loss = cudalize(Variable(torch.tensor(0.0)))
    for line_num in range(line_number):
        lg_title_tensor_variable = cudalize(Variable(title_tensor)) # torch.Size([10, 9746])
        lg_genre_variable = cudalize(Variable(genre_embedding[genre_tensor])) # torch.Size([10, 3])
        lg_input = torch.cat((lyric_latent_variable, lg_title_tensor_variable, lg_genre_variable), 1) # torch.Size([10, 10261])

        end_output, topic_output, lg_hidden = lyric_generator(lg_input, lg_hidden, batch_size)
        lg_end_outputs[line_num] = end_output

        sg_hidden = topic_output.view(1, batch_size, -1) # torch.Size([1, 10, 512])
        sg_word_tensor = torch.from_numpy(np.array([word_embedding[SOS]]*batch_size)).type(torch.FloatTensor) # torch.Size([10, 9746])
        sg_word_outputs = cudalize(Variable(torch.zeros(line_length-1, batch_size, sentence_generator.output_size))) # torch.Size([19, 10, 9746])
        for line_idx in range(1, line_length):
            # title_tensor - this line
            sg_genre_tensor = genre_embedding[genre_tensor] # torch.Size([10, 3])
            sg_input = torch.cat((sg_word_tensor, title_tensor, sg_genre_tensor), 1) # torch.Size([10, 19495])
            sg_input = cudalize(Variable(sg_input))

            sg_output, sg_hidden = sentence_generator(sg_input, sg_hidden, batch_size)
            sg_word_outputs[line_idx-1] = sg_output

            _, topi = softmax(sg_output).topk(1)
            ni = topi.cpu().view(-1) # workable, but be careful
            sg_word_tensor = torch.from_numpy(word_embedding[ni]).type(torch.FloatTensor)
        
        sg_logits = sg_word_outputs.transpose(0, 1).contiguous() # -> batch x seq, torch.Size([10, 21, 9746])
        sg_target = cudalize(lyric_tensor[:,line_num,1:line_length].contiguous()) # -> batch x seq, torch.Size([10, 21])
        sg_length = line_length_tensor[:, line_num] - 1
        sg_length[sg_length<0] = 0

        if sg_length.sum().item() == 0:
            continue

        try:
            sg_word_loss += masked_cross_entropy(sg_logits, sg_target, sg_length)
        except:
            pdb.set_trace()
            # sg_word_loss = masked_cross_entropy(sg_logits, sg_target, sg_length)
        if torch.isnan(sg_word_loss).item() == 1:
            pdb.set_trace()
    
    lg_logits = lg_end_outputs.transpose(0, 1).contiguous() # -> batch x seq, torch.Size([10, 40, 2])
    lg_target = cudalize(line_end_embedding[line_num_tensor-1][:,:line_number].contiguous()) # -> batch x seq, torch.Size([10, 40])
    lg_length = line_num_tensor

    try:
        lg_end_loss = masked_cross_entropy(lg_logits, lg_target, lg_length)
    except:
        pdb.set_trace()
    
    sg_word_loss /= line_number

    if torch.isnan(sg_word_loss).item() == 1:
        pdb.set_trace()

    sg_word_loss = sg_word_loss_weight*sg_word_loss
    lg_end_loss = lg_end_loss_weight*lg_end_loss
    auto_loss = lg_end_loss + sg_word_loss

    sg_word_loss_data = sg_word_loss.item()
    lg_end_loss_data = lg_end_loss.item()
    auto_loss_data = auto_loss.item()

    # pdb.set_trace()

    if model_type == 'train':
        auto_loss.backward()
        sentence_encoder_optimizer.step()
        lyric_encoder_optimizer.step()
        lyric_generator_optimizer.step()
        sentence_generator_optimizer.step()
    
    # print(auto_loss_data, sg_word_loss_data, lg_end_loss_data)
    return auto_loss_data, sg_word_loss_data, lg_end_loss_data

def trainEpochs(sentence_encoder, 
                lyric_encoder, 
                lyric_generator, 
                sentence_generator, 
                batch_size, 
                learning_rate, 
                num_epoch, 
                print_every,
                saving_dir = SavingDir):
    sentence_encoder_optimizer = torch.optim.Adam(sentence_encoder.parameters(), lr=learning_rate)
    lyric_encoder_optimizer = torch.optim.Adam(lyric_encoder.parameters(), lr=learning_rate)
    lyric_generator_optimizer = torch.optim.Adam(lyric_generator.parameters(), lr=learning_rate)
    sentence_generator_optimizer = torch.optim.Adam(sentence_generator.parameters(), lr=learning_rate)

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

        print_loss_total_auto = 0.0  # Reset every print_every
        print_loss_total_auto_list = []
        print_loss_total_word = 0.0  # Reset every print_every
        print_loss_total_word_list = []
        print_loss_total_end = 0.0  # Reset every print_every
        print_loss_total_end_list = []

        for batch, data in enumerate(train_loader, 0):
            title_tensor = data['title'].type(torch.FloatTensor) # torch.Size([10, 9746])
            genre_tensor = data['genre'] # torch.Size([10]), tensor([0, 2, 1, 1, 0, 1, 2, 1, 1, 1])
            lyric_tensor = data['lyric'] # torch.Size([10, 40, 32])
            line_length_tensor = data['line_length'] # torch.Size([10, 40])
            line_num_tensor = data['line_numb'] # torch.Size([10]), tensor([40, 17, 31, 38, 40, 40, 22,  9, 12, 39])

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
                                                       sentence_encoder_optimizer,
                                                       lyric_encoder_optimizer,
                                                       lyric_generator_optimizer,
                                                       sentence_generator_optimizer,
                                                       len(line_num_tensor))
        
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
        
        validation_loss_auto_list = []
        validation_loss_word_list = []
        validation_loss_end_list = []

        for _, val_data in enumerate(val_loader, 0):
            title_tensor = val_data['title'].type(torch.FloatTensor) # torch.Size([10, 9746])
            genre_tensor = val_data['genre'] # torch.Size([10]), tensor([0, 2, 1, 1, 0, 1, 2, 1, 1, 1])
            lyric_tensor = val_data['lyric'] # torch.Size([10, 40, 32])
            line_length_tensor = val_data['line_length'] # torch.Size([10, 40])
            line_num_tensor = val_data['line_numb'] # torch.Size([10]), tensor([40, 17, 31, 38, 40, 40, 22,  9, 12, 39])

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
        
        # write to tensorboard
        iter_epoch += 1
        writer.add_scalars(saving_dir+'/auto_loss/train_val_epoch', {'train': print_loss_auto_avg_train, 'val': print_loss_auto_avg_val}, iter_epoch)
        writer.add_scalars(saving_dir+'/word_loss/train_val_epoch', {'train': print_loss_word_avg_train, 'val': print_loss_word_avg_val}, iter_epoch)
        writer.add_scalars(saving_dir+'/end_loss/train_val_epoch', {'train': print_loss_end_avg_train, 'val': print_loss_end_avg_val}, iter_epoch)

        # # save models    
        # torch.save(sentence_encoder.state_dict(), saving_dir+'/sentence_encoder_'+str(epoch+1))
        # torch.save(lyric_encoder.state_dict(), saving_dir+'/lyric_encoder_'+str(epoch+1))
        # torch.save(lyric_generator.state_dict(), saving_dir+'/lyric_generator_'+str(epoch+1))
        # torch.save(sentence_generator.state_dict(), saving_dir+'/sentence_generator_'+str(epoch+1))
        
if __name__=='__main__':
    word_embedding_size = DictionarySize
    title_embedding_size = TitleSize
    genre_embedding_size = GenreSize

    # sentence encoder - se
    se_input_size = word_embedding_size + title_embedding_size + genre_embedding_size
    se_embedding_size = 300
    se_hidden_size = 300 # 512
    sentence_encoder = SentenceEncoder(se_input_size, se_embedding_size, se_hidden_size)
    sentence_encoder = cudalize(sentence_encoder)
    sentence_encoder.train()

    # lyric encoder - le
    le_input_size = se_hidden_size + title_embedding_size + genre_embedding_size
    le_embedding_size = 300
    le_hidden_size = 512
    lyric_encoder = LyricEncoder(le_input_size, le_embedding_size, le_hidden_size)
    lyric_encoder = cudalize(lyric_encoder)
    lyric_encoder.train()

    # lyric generator - lg
    lg_input_size = le_hidden_size + title_embedding_size + genre_embedding_size
    lg_embedding_size = 300
    lg_hidden_size = 300 # 512
    lg_topic_latent_size = 300 # 512
    lg_topic_output_size = 300 # 512
    lyric_generator = LyricGenerator(lg_input_size, lg_embedding_size, lg_hidden_size, lg_topic_latent_size, lg_topic_output_size)
    lyric_generator = cudalize(lyric_generator)
    lyric_generator.train()

    # sentence generator - sg
    sg_input_size = word_embedding_size + title_embedding_size + genre_embedding_size
    sg_embedding_size = 300
    sg_hidden_size = 300 # 512
    sg_output_size = DictionarySize
    sentence_generator = SentenceGenerator(sg_input_size, sg_embedding_size, sg_hidden_size, sg_output_size)
    sentence_generator = cudalize(sentence_generator)
    sentence_generator.train()

    batch_size = 20 # 20
    learning_rate = LearningRate
    num_epoch = 500
    print_every = 1
    
    trainEpochs(sentence_encoder, lyric_encoder, lyric_generator, sentence_generator, batch_size, learning_rate, num_epoch, print_every)
    writer.close()
