import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.nn import functional

# ------------------------------------------------------------------------------
use_cuda = torch.cuda.is_available()
print ('Using cuda: ', use_cuda)

def cudalize(item, use_cuda=use_cuda):
    if use_cuda:
        return item.cuda()
    else:
        return item
# ------------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SentenceEncoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size):
        super(SentenceEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Linear(input_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size)

    def forward(self, input, hidden, batch_size):
        output = self.embedding(input).view(1, batch_size, -1)
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

class LyricEncoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size):
        super(LyricEncoder, self).__init__()
        self.hidden_size = hidden_size
        # self.embedding = nn.Linear(input_size, embedding_size)
        # self.gru = nn.GRU(embedding_size, hidden_size)
        self.gru = nn.GRU(input_size, hidden_size)

    def forward(self, input, hidden, batch_size):
        # output = self.embedding(input).view(1, batch_size, -1)
        output = input.view(1, batch_size, -1)
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

class LyricGenerator(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, topic_latent_size, topic_output_size):
        super(LyricGenerator, self).__init__()
        self.hidden_size = hidden_size
        # self.embedding = nn.Linear(input_size, embedding_size)
        # self.gru = nn.GRU(embedding_size, hidden_size)
        self.gru = nn.GRU(input_size, hidden_size)
        self.end_layer = nn.Linear(hidden_size, 2)
        # self.softmax = nn.LogSoftmax(dim=1)
        self.topic_layer_1 = nn.Linear(hidden_size, topic_latent_size)
        self.topic_layer_relu = nn.ReLU(inplace=True)
        self.topic_layer_2 = nn.Linear(topic_latent_size, topic_output_size)
        
    def forward(self, input, hidden, batch_size):
        # output = self.embedding(input).view(1, batch_size, -1)
        output = input.view(1, batch_size, -1)
        output, hidden = self.gru(output, hidden)
        end_output = self.end_layer(output[0])
        # output = self.softmax(self.out(output[0]))
        topic_output = self.topic_layer_1(output[0])
        topic_output = self.topic_layer_relu(topic_output)
        topic_output = self.topic_layer_2(topic_output)
        return end_output, topic_output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

class SentenceGenerator(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size):
        super(SentenceGenerator, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding = nn.Linear(input_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, batch_size):
        output = self.embedding(input).view(1, batch_size, -1)
        output, hidden = self.gru(output, hidden)
        output = self.out(output[0])
        # output = self.softmax(output)
        return output, hidden
    
    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

class LyricDiscriminator(nn.Module):
    def __init__(self, input_size):
        super(LyricDiscriminator, self).__init__()
        latent_size = input_size//2
        self.fc_1 = nn.Linear(input_size, latent_size)
        self.lrelu_1 = nn.LeakyReLU(0.01)
        # self.fc_2 = nn.Linear(1024, 256)
        # self.lrelu_2 = nn.LeakyReLU(0.01)
        self.fc_3 = nn.Linear(latent_size, 1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        output = self.lrelu_1(self.fc_1(input))
        # output = elf.lrelu_2(self.fc_2(output))
        # output = self.sigmoid(self.fc_3(output))
        output = self.fc_3(output)
        return output
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class LyricClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(LyricClassifier, self).__init__()
        latent_size = input_size//2
        self.fc_1 = nn.Linear(input_size, latent_size)
        self.relu_1 = nn.ReLU()
        self.fc_2 = nn.Linear(latent_size, output_size)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        output = self.relu_1(self.fc_1(input))
        # output = self.sigmoid(self.fc_2(output))
        output = self.fc_2(output)
        return output
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

# ------------------------------------------------------------------------------
# masked_cross_entropy
# The following code is downloaded from
# https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/masked_cross_entropy.py
def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand

def masked_cross_entropy(logits, target, length):
    length = cudalize(Variable(torch.LongTensor(length)))
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """
    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = functional.log_softmax(logits_flat, dim=1)
    # pdb.set_trace()
    # log_probs_flat = functional.log_softmax(logits_flat, dim=1)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    return loss

def masked_cross_entropy_nounk(logits, target, length, nounk_mask_tensor):
    length = cudalize(Variable(torch.LongTensor(length)))
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """
    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = functional.log_softmax(logits_flat, dim=1)
    # pdb.set_trace()
    # log_probs_flat = functional.log_softmax(logits_flat, dim=1)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))

    # import pdb; pdb.set_trace()

    losses = losses * mask.float() * nounk_mask_tensor.float()
    loss = losses.sum() / nounk_mask_tensor.float().sum()
    return loss