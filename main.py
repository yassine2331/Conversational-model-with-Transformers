'''
Template for the 4th assignment
Student: YASSINE OUESLATI
'''

############################
# Packages
############################
import torch
import torch.nn as nn
import numpy as np
import math
import regex as re
import ast
import json
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
import torch.optim as optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from torch.nn import TransformerDecoder, TransformerDecoderLayer
from accelerate import Accelerator
from accelerate.utils import GradientAccumulationPlugin
import matplotlib.pyplot as plt
############################
# Classes
############################
# Vocabulary class
class Vocabulary:
    '''
    Class for dealing with our corpus
    '''

    def __init__(self, name, pairs):
        """
        Args:
            name (str): name of the language
            pairs (list): list of pairs of sentences
        """
        self.name = name
        self.word2index = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2}
        self.index2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>"}
        self.pairs = pairs
        self.word_count = {}
        self.tokenized_pairs= []
    def add_word(self, word):
        '''
        Add a word to the vocabulary
        :param word: a string
        '''
        # TODO: add the word to the vocabulary
        id = len(self.index2word.values())
        if (word not in self.index2word.values()) and (word not in self.word2index.keys()):
            self.word2index.update({word:id})
            self.index2word.update({id:word})
        if word in self.word_count.keys():
            self.word_count[word] += 1
        else:
            self.word_count.update({word: 1})
        return id

    def add_sentence(self, sentence):
        '''
        Add a sentence to the vocabulary
        :param sentence: list of strings (words)
        '''
        # TODO add the sentence to the vocabulary, this method will call the add_word method
        # TODO fix the punctuation function

        for word in sentence:
            self.add_word(word)
        return sentence


    def len_pairs(self):
        return len(self.pairs)


    def pair(self,index):
        return self.pairs[index]


    def tokenize(self):

        for s1 , s2 in tqdm(self.pairs,desc="tokenize"):
            s1 = clear_punctuation(s1).split()
            s2 = clear_punctuation(s2).split()
            self.add_sentence(s1)
            self.add_sentence(s2)
            self.tokenized_pairs.append((s1+["<EOS>"],["<SOS>"]+s2+["<EOS>"]))
        return 0

    def tokenized_pair(self,index):
        s1 ,s2= self.pair(index)
        s1 = clear_punctuation(s1).split()
        s2 = clear_punctuation(s2).split()
        tokenized_question = []
        tokenized_answer = [self.word2index["<SOS>"]]
        for word in s1:
            tokenized_question.append(self.word2index[word])
        tokenized_question.append(self.word2index["<EOS>"])

        for word in s2:
            tokenized_answer.append(self.word2index[word])


        return tokenized_question,tokenized_answer

    def untokenized_pair(self,index):
        s1, s2 = self.pair(index)
        s1 = clear_punctuation(s1).split()
        s2 = clear_punctuation(s2).split()
        s1.append("<EOS>")
        s2 = ["<SOS>"]+ s2+ ["<EOS>"]
        return s1, s2


    def tokenize_sentence(self,sentance):
        sentance = clear_punctuation(sentance).split()
        tokenized_sentence = []
        for word in sentance:
            if word in self.word2index.keys():
                tokenized_sentence.append(self.word2index[word])
            else:
                tokenized_sentence.append(self.word2index["<PAD>"])

        return tokenized_sentence + [self.word2index["<EOS>"]]

    def untokenize_sentence(self,array):

        sentence = []
        for id in array:
            sentence.append(self.index2word[id])
        return sentence
    def len_pair(self,index):
        pair = self.tokenized_pair(index)

        return max(len(pair[0]),len(pair[1]))

    def plot_freqeuncy_pairs(self,file_name):
        x = []

        for index in tqdm(range(len(self.pairs))):
            x.append(self.len_pair(index))

        plt.hist(x)
        plt.show()

        df = pd.DataFrame(x, columns=['maximum length per pair of sentences'])
        fig = sns.displot(df,x="maximum length per pair of sentences",discrete=True)
        fig.savefig(fname=file_name)


    def Remove_pairs(self,max_length):

        index = 0
        for _ in tqdm(range(len(self.pairs)),desc="remove pairs iwth max len"):
            if self.len_pair(index)>max_length:
                self.pairs.pop(index)

            else:
                index+=1



    def remove_word(self,word):


        removal_indices = dict()
        for index, (p1, p2) in enumerate(self.tokenized_pairs):
            if word in p1 or word in p2:

                removal_indices.update({index:index})

        # Remove all marked pairs in a single operation
        #self.pairs = [pair for i, pair in enumerate(self.pairs) if i not in removal_indices]
        #self.tokenized_pairs= [pair for i, pair in enumerate(self.tokenized_pairs) if i not in removal_indices]
        return removal_indices
    def remove_pairs_word(self,min_freq):

        low_freq_words = {word for word, count in self.word_count.items() if count <= min_freq}

        removal_indices= dict()

        self.tokenized_pairs =[(clear_punctuation(p1).split()+["<EOS>"],["<SOS>"]+clear_punctuation(p2).split()+["<EOS>"]) for p1,p2 in tqdm(self.pairs)]

        for word in tqdm(low_freq_words,desc="remove min freq word"):
            indices =self.remove_word(word)
            removal_indices.update(indices)

        #self.tokenized_pairs = [(clear_punctuation(p1).split() + ["<EOS>"], ["<SOS>"] + clear_punctuation(p2).split() + ["<EOS>"]) for p1, p2 in tqdm(self.pairs)]

        self.pairs = [pair for i, pair in enumerate(self.pairs) if i not in removal_indices.values()]
        self.tokenized_pairs = [(clear_punctuation(p1).split() + ["<EOS>"], ["<SOS>"] + clear_punctuation(p2).split() + ["<EOS>"]) for p1, p2 in tqdm(self.pairs)]

        #self.tokenized_pairs= [pair for i, pair in enumerate(self.tokenized_pairs) if i not in removal_indices.values()]



    def Count_words(self):

        for pair in  tqdm(self.pairs,desc="counting words"):
            pair1 = clear_punctuation(pair[0]).split()
            pair2 = clear_punctuation(pair[1]).split()
            for word in pair1:
                if word in self.word_count.keys():
                    self.word_count[word] +=1
                else:
                    self.word_count.update({word,1})
            for word in pair2:
                if word in self.word_count.keys():
                    self.word_count[word] +=1
                else:
                    self.word_count.update({word,1})







def clear_punctuation(s):
    '''
    This function removes all the punctuation from a sentence and insert a blank between any letter and !?.
    :param s: a string
    :return: the "cleaned" string
    '''
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)  # Remove all the character that are not letters, puntuation or numbers
    # Insert a blank between any letter and !?. using regex

    s = re.sub(r"([a-zA-Z])([!?.])", r"\1 \2", s)
    s = re.sub(r"([!?.])([a-zA-Z])", r"\1 \2", s)

    return s.lower()


# Dataset class
class Dataset(torch.utils.data.Dataset):

    def __init__(self, vocabulary, pairs):
        # TODO We want vocabulary and pairs to be attributes of the class

        self.data = []
        self.eos = vocabulary.word2index["<EOS>"]
        for index in range(len(vocabulary.tokenized_pairs)):
            self.data.append(vocabulary.tokenized_pair(index))



    def __len__(self):
        # TODO how many pairs do we have?
        return len(self.data)


    def __getitem__(self, ix):
        # TODO returns two tensors (question, answer) of the pair at index ix
        # TODO the tensors should be of type torch.tensor and should contain integers (word indices)
        x = self.data[ix][0]
        y = self.data[ix][1]
        z = y[1:]+[self.eos]
        return torch.tensor(x), torch.tensor(y),torch.tensor(z)



def collate_fn(batch, pad_value):
  data, targets,target2 = zip(*batch)

  padded_data = nn.utils.rnn.pad_sequence(data, batch_first=True,padding_value=pad_value)
  padded_targets = nn.utils.rnn.pad_sequence(targets, batch_first=True,padding_value=pad_value)
  padded_targets2 = nn.utils.rnn.pad_sequence(target2, batch_first=True,padding_value=pad_value)
  return padded_data, padded_targets,padded_targets2




class PositionalEncoding(nn.Module):
    '''
    Adapted from
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    '''

    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        try:
            assert x.size(0) < self.max_len
        except:
            print(
                "The length of the sequence is bigger than the max_len of the positional encoding. Increase the max_len or provide a shorter sequence.")
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, pad_id=0, encoder_layers=6, decoder_layers=6, dim_feedforward=2048 ,
                 num_heads=8, dropout_p=0.1):
        super().__init__()
        # TODO add an embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model,padding_idx=pad_id)
        # TODO add a positional encoding layer
        self.pos_encoder = PositionalEncoding(d_model, dropout_p)
        # TODO add a transformer layer, you can use nn.Transformer. You can use the default values for the parameters, but what about batch_first?
        self.transformer = nn.Transformer(d_model=d_model,
                                         nhead=num_heads,
                                         num_encoder_layers=encoder_layers,
                                         num_decoder_layers=decoder_layers,
                                         dropout=dropout_p,
                                          dim_feedforward=dim_feedforward,
                                          batch_first=True)
        # TODO add a linear layer. Note: output should be probability distribution over the vocabulary
        self.linear = nn.Linear(d_model,vocab_size)

        # Stuff you may need
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.num_heads = num_heads

    def create_padding_mask(self, x, pad_id=0):
        # TODO create a boolean mask for the <PAD> tokens

        return x == pad_id

    def forward(self, src, tgt):
        # S is the source sequence length, T is the target sequence length, N is the batch size, E is the feature number
        # src: (N, S)
        # tgt: (N, T)
        # src_pad_mask: (N, S)
        # tgt_pad_mask: (N, T)
        # mask the future : (N * num_heads, T, T)

        src_pad_mask = self.create_padding_mask(src, self.pad_id)  # (N, S)
        tgt_pad_mask = self.create_padding_mask(tgt, self.pad_id)  # (N, T)

        src = self.embedding(src)
        tgt = self.embedding(tgt)

        src = self.pos_encoder(src)  # (N, S, E)
        tgt = self.pos_encoder(tgt)  # (N, T, E)

        # Mask the memory
        memory_key_padding_mask = src_pad_mask  # (N, S)

        # Mask the future
        #tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1), dtype=torch.bool).to(tgt.device)  # (T, T)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).bool().to(tgt.device)  # (T, T)
        #tgt_mask = tgt_mask

        # Expand to make it N * num_heads, T, T
        #tgt_mask = tgt_mask.unsqueeze(0).repeat(tgt.size(0) * self.num_heads, 1, 1)
        # (N, T, T)
        # Transformer
        output = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask,
                                  tgt_key_padding_mask=tgt_pad_mask,
                                  memory_key_padding_mask=memory_key_padding_mask)  # (N, T, E)
        # Linear layer
        output = self.linear(output)  # (N, T, V)
        return output





class Corpus:

    corpus = dict()
    convo = []

    def __init__(self,convo = []):
        self.convo = convo
        self.corpus = dict()

    def add_sentance(self,id,sentance):
        self.corpus.update({id:sentance})
    def add_convo(self,convo):
        self.convo.append(convo)

    #how many conversations are in this class
    def len_convo(self):
        return len(self.convo)

    def len_corpus(self):
        return len(self.corpus)

    def get_convo(self,i):
        return [self.corpus[sentance] for sentance in self.convo[i]]

    def get_sentace(self,key):
        return self.corpus[key]

    def get_pair(self,i):
        convo = self.get_convo(i)
        return [(convo[j],convo[j+1]) for j in range(len(convo)-1)]




class TransformerSeparate(nn.Module):
    def __init__(self, vocab_size, d_model=512, pad_id=0, encoder_layers=6, decoder_layers=6, dim_feedforward=2048, num_heads=8, dropout_p=0.1):
        super().__init__()

        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.d_model = d_model

        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)

        # Positional Encoding Layer
        self.pos_encoder = PositionalEncoding(d_model, dropout_p)

        # Transformer Encoder Layer
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout_p, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=encoder_layers)

        # Transformer Decoder Layer
        decoder_layer = TransformerDecoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout_p, batch_first=True)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=decoder_layers)

        # Linear Layer
        self.linear = nn.Linear(d_model, vocab_size)
        #hmm
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.num_heads = num_heads

    def create_padding_mask(self, x):
        return x == self.pad_id

    def forward(self, src, tgt):
        # S is the source sequence length, T is the target sequence length, N is the batch size, E is the feature number
        # src: (N, S)
        # tgt: (N, T)
        # src_pad_mask: (N, S)
        # tgt_pad_mask: (N, T)
        # mask the future : (N * num_heads, T, T)
        src_pad_mask = self.create_padding_mask(src)
        tgt_pad_mask = self.create_padding_mask(tgt)

        # Embedding and Positional Encoding
        src = self.embedding(src)
        tgt = self.embedding(tgt)

        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)

        # Memory Key Padding Mask
        memory_key_padding_mask = src_pad_mask


        # Transformer Encoder
        memory = self.transformer_encoder(src, src_key_padding_mask=src_pad_mask)

        # Transformer Decoder
        output = self.transformer_decoder(tgt, memory, tgt_key_padding_mask=tgt_pad_mask,memory_key_padding_mask=memory_key_padding_mask)

        # Linear Layer
        output = self.linear(output)

        return output





############################
# Training
############################


def train(model,train_data,criterion,optimizer,scheduler,epoch,vocabulary,DEVICE):

    num_batches = len(train_data)


    model.train()  # turn on train mode

    total_loss = 0.
    log_interval = 150


    start_time = time.time()

    batch =0
    final_loss=0

    for x , t1, t2 in train_data:


        x = x.to(DEVICE)
        t1 = t1.to(DEVICE)
        t2  = t2.to(DEVICE)
        tt= t2
        output = model(x,t1)

        output= output.to(DEVICE)

        output_flat = output.view(-1,model.vocab_size)
        t2 = t2.view(-1)
        loss = criterion(output_flat, t2)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.9)
        optimizer.step()

        total_loss += loss.item()
        final_loss += loss.item()

        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:04.4f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()

            print("| Question: ", [vocabulary.index2word[idx.item()] for idx in x[0] ])
            # Print the words of the answers
            print("| Answer t2: ", [vocabulary.index2word[idx.item()] for idx in tt[0]])
            #print predinction 2 ??

            print("| Answer t1: ", [vocabulary.index2word[idx.item()] for idx in t1[0] ])
            # Print the words of the prediction
            print("| Prediction: ", [vocabulary.index2word[idx.item()] for idx in torch.argmax(output[0], dim=-1)])



        batch = batch + 1

    print(final_loss/num_batches)
    return final_loss/num_batches






def train_ga(model,train_data,criterion,optimizer,scheduler,epoch,vocabulary,DEVICE):

    num_batches = len(train_data)


    model.train()  # turn on train mode

    total_loss = 0.
    log_interval = 5000

    accum_iter = 32

    start_time = time.time()

    batch =0
    final_loss=0

    for x , t1, t2 in train_data:


        x = x.to(DEVICE)
        t1 = t1.to(DEVICE)
        t2  = t2.to(DEVICE)
        tt= t2
        output = model(x,t1)

        output= output.to(DEVICE)

        output_flat = output.view(-1,model.vocab_size)
        t2 = t2.view(-1)
        loss = criterion(output_flat, t2)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.9)




        total_loss += loss.item()
        final_loss += loss.item()


        loss = loss / accum_iter
        loss.backward()
        # weights update only every accum_iter step
        if ((batch + 1) % accum_iter == 0) or (batch + 1 == num_batches):
            optimizer.step()
            optimizer.zero_grad()



        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:04.4f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()

            print("| Question: ", [vocabulary.index2word[idx.item()] for idx in x[0] if idx.item()!=0])
            # Print the words of the answers
            print("| Answer t2: ", [vocabulary.index2word[idx.item()] for idx in tt[0]if idx.item()!=0])
            #print predinction 2 ??

            print("| Answer t1: ", [vocabulary.index2word[idx.item()] for idx in t1[0]if idx.item()!=0])
            # Print the words of the prediction
            print("| Prediction: ", [vocabulary.index2word[idx.item()] for idx in torch.argmax(output[0], dim=-1)])



        batch = batch + 1

    print(final_loss/num_batches)
    return final_loss/num_batches







def train_ga_hf(model,train_data,criterion,optimizer,scheduler,epoch,vocabulary,accelerator,DEVICE):

    num_batches = len(train_data)


    model.train()  # turn on train mode

    total_loss = 0.
    log_interval = 1000

    accum_iter = 32

    start_time = time.time()

    batch =0
    final_loss=0

    for x , t1, t2 in train_data:
        with accelerator.accumulate(model):
            tt = t2
            output = model(x,t1)
            output= output.to(DEVICE)

            output_flat = output.view(-1,model.vocab_size)
            t2 = t2.view(-1)

            loss = criterion(output_flat, t2)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.9)


            total_loss += loss.item()
            final_loss += loss.item()

            accelerator.backward(loss)

            optimizer.step()
            optimizer.zero_grad()



        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:04.4f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()

            print("| Question: ", [vocabulary.index2word[idx.item()] for idx in x[0] if idx.item()!=0])
            # Print the words of the answers
            print("| Answer t2: ", [vocabulary.index2word[idx.item()] for idx in tt[0]if idx.item()!=0])
            #print predinction 2 ??

            print("| Answer t1: ", [vocabulary.index2word[idx.item()] for idx in t1[0]if idx.item()!=0])
            # Print the words of the prediction
            print("| Prediction: ", [vocabulary.index2word[idx.item()] for idx in torch.argmax(output[0], dim=-1)])



        batch = batch + 1

    print(final_loss/num_batches)
    return final_loss/num_batches






def evaluate(model,eval_data,criterion,DEVICE):
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    with torch.no_grad():
        for x , t1, t2 in eval_data:

            x = x.to(DEVICE)
            t1 = t1.to(DEVICE)
            t2  = t2.to(DEVICE)
            output = model(x,t1)
            output= output.to(DEVICE)
            output_flat = output.view(-1,model.vocab_size)
            t2 = t2.view(-1)
            total_loss +=  criterion(output_flat, t2).item()

    print("evaluate loss = ",total_loss / (len(eval_data) - 1))
    return total_loss / (len(eval_data) - 1)




############################
# Training
############################


############################
# Methods
############################


def inspect_files():

    #initialising paths
    path1="Data/movie_conversations.txt"
    path2 = "Data/movie_lines.txt"

    #init the corpus
    corpus = Corpus()

    #deal withe the firt text
    with open(path1,mode="r")as f:
        text_1=f.read()

    array = text_1.split("\n")

    corpus = Corpus()

    for i in range(len(array)):

        x = ast.literal_eval(array[i].split(" +++$+++ ")[-1])
        corpus.add_convo(x)

    #deal withe second text
    with open(path2,mode="rb")as f:
        text=f.read()

    text_2 = str(text)[2:-1]


    array = text_2.replace("\\n","\n").replace("\\'","\'").split("\n")

    for i in range(len(array)-1):
        x = array[i].split(" +++$+++ ")
        corpus.add_sentance(x[0],x[-1])


    return corpus


def create_list_pairs(corpus:Corpus):
    #input:corpus
    #return: list of pairs

    list_pairs = []

    for i in range(corpus.len_convo()):
        for pair in corpus.get_pair(i):
            list_pairs.append(pair)

    return list_pairs





def save_data(array,name ='data.pkl'):

    with open(name, 'wb') as f:
        pickle.dump(array, f)
    return 0


def load_data(name='data.pkl'):
    with open(name, 'rb') as f:
        array = pickle.load(f)
    return array


def plot_frequency(array,file_name):
    array.sort()

    plt.hist(array[29491:])
    plt.show()
    """
    print(array)
    df = pd.DataFrame(array, columns=['words count'])
    fig = sns.displot(df, x="words count", discrete=True)
    fig.savefig(fname=file_name)
    """
    return 0


def sample_without_replacement(tensor, n):


    indices = torch.randperm(tensor.numel())[:n]
    sampled_elements = tensor.view(-1)[indices]

    return sampled_elements





def generate_output(model, input_sentence, vocabulary, max_length=50):

    model.eval()  # Set the model to evaluation mode

    # Tokenize the input sentence
    tokenized_input = vocabulary.tokenize_sentence(input_sentence)

    # Convert to tensor and add batch dimension
    input_tensor = torch.tensor([tokenized_input]).to(DEVICE)

    # Prepare the initial target tensor with <SOS>
    sos_token_id = vocabulary.word2index["<SOS>"]
    target_tensor = torch.tensor([[sos_token_id]]).to(DEVICE)

    # Generate the output sequence
    for _ in range(max_length):
        with torch.no_grad():
            output = model(input_tensor, target_tensor)
            next_token_id = output[0, -1, :].argmax().item()

        # Append the predicted token id to the target sequence
        target_tensor = torch.cat([target_tensor, torch.tensor([[next_token_id]]).to(DEVICE)], dim=-1)

        # Break if <EOS> token is generated
        if next_token_id == vocabulary.word2index["<EOS>"]:
            break

    # Convert the output tokens to words
    output_sequence = [vocabulary.index2word[token_id] for token_id in target_tensor[0].cpu().numpy()]

    # Join the words to form the output sentence
    output_sentence = ' '.join(output_sequence[1:])  # Skip the <SOS> token

    return output_sentence








def generate_output_top_k(model, input_sentence, vocabulary, max_length=50, top_k=30):

    model.eval()  # Set the model to evaluation mode

    # Tokenize the input sentence
    tokenized_input = vocabulary.tokenize_sentence(input_sentence)

    # Convert to tensor and add batch dimension
    input_tensor = torch.tensor([tokenized_input]).to(DEVICE)

    # Prepare the initial target tensor with <SOS>
    sos_token_id = vocabulary.word2index["<SOS>"]
    target_tensor = torch.tensor([[sos_token_id]]).to(DEVICE)

    # Generate the output sequence
    for _ in range(max_length):
        with torch.no_grad():
            output = model(input_tensor, target_tensor)



            # Get the top k tokens and their probabilities
            top_k_probs, top_k_indices = torch.topk(output[0, -1, :], top_k,dim=-1, sorted=True)

            top_k_probs = torch.softmax(top_k_probs,dim=-1)

            # Sample from the top k tokens

            next_token_id = np.random.choice(top_k_indices.cpu().numpy(), p=top_k_probs.cpu().numpy())

        target_tensor = torch.cat([target_tensor, torch.tensor([[next_token_id]]).to(DEVICE)], dim=-1)

        # Break if <EOS> token is generated
        if next_token_id == vocabulary.word2index["<EOS>"]:
            break

    # Convert the output tokens to words
    output_sequence = [vocabulary.index2word[token_id] for token_id in target_tensor[0].cpu().numpy()]

    # Join the words to form the output sentence
    output_sentence = ' '.join(output_sequence[1:])  # Skip the <SOS> token

    return output_sentence



def plot_losses(train_loss, eval_loss, title='Training and Evaluation Loss', xlabel='Epochs', ylabel='Loss'):

    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, marker='o', linestyle='-', color='blue', label='Training Loss')
    plt.plot(eval_loss, marker='s', linestyle='-', color='red', label='Evaluation Loss')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('my_plot.png')



def prepare_data():
    # Download the data
    corpus = Corpus()
    corpus = inspect_files()

    # Create the pairs

    vocab = Vocabulary(name="eng",pairs=create_list_pairs(corpus))

    # Tokenize the data
    vocab = load_data()

    vocab.tokenize()   # For now

    print(vocab.pair(0))
    print(vocab.untokenized_pair(0))
    print(vocab.tokenized_pair(0))
    print(vocab.tokenize_sentence("hey there!"))



    # Filter out the sentences that are too long
    vocab.plot_freqeuncy_pairs(file_name = "before_change.png")

    vocab.Remove_pairs(20)

    vocab.plot_freqeuncy_pairs(file_name = "after_change.png")
    print(len(vocab.pairs))

    save_data(vocab)
    # Filter out the words that are too rare

    #print(vocab.word_count.values())
    plot_frequency(list(vocab.word_count.values()), "word_count.png")
    print(len(vocab.pairs))

    vocab.remove_pairs_word(50)



    # SAVE and put the code above into a function that you will call if you need to generate something slightly different


    return vocab





if __name__ == "__main__":
    # !!! Don't change the seed !!!
    torch.manual_seed(42)
    # !!!!!!

    vocab = prepare_data()

    n= len(vocab.pairs)
    save_data(vocab)
    tensor = torch.arange(0, n)

    sampled_tensor = sample_without_replacement(tensor, 10000)

    new_pairs = [vocab.pairs[i] for i in sampled_tensor]

    print(len(new_pairs))

    vocab_data = Vocabulary(name="data", pairs=new_pairs)

    vocab_data.tokenize()

    print(vocab_data.tokenized_pairs)

    save_data(vocab_data,"new_data.pkl")
    print(vocab_data.tokenized_pair(3))
    #data set

    vocab_data = vocab


    batch_size = 64
    dataset = Dataset(vocab_data, vocab_data.pairs)

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [16000, 4000])


    print(len(vocab_data.pairs))

    if batch_size == 1:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=lambda b: collate_fn(b, vocab_data.word2index["<PAD>"]), shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=lambda b: collate_fn(b, vocab_data.word2index["<PAD>"]), shuffle=True)



    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    print("DEVICE: ", DEVICE)



    lr = 0.0001

    model = TransformerSeparate(vocab_size=len(vocab_data.word2index),dropout_p=0.3)


    model = model.to(DEVICE)



    criterion = nn.CrossEntropyLoss(ignore_index=vocab_data.word2index["<PAD>"])

    optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=0.9,step_size=3)


    traing_losses = []

    eval_losses = []

    for epoch in range(1,7):
        training_loss = train(model, train_dataloader, criterion,optimizer,scheduler, epoch,vocab_data, DEVICE)
        scheduler.step()
        eval_loss = evaluate(model,val_dataloader,criterion,DEVICE)

        traing_losses.append(training_loss)
        eval_losses.append(eval_loss)

        plot_losses(traing_losses, eval_losses)






    accelerator = Accelerator(gradient_accumulation_steps=32)

    lr = 0.0001

    model = TransformerModel(vocab_size=len(vocab_data.word2index),dropout_p=0.3)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab_data.word2index["<PAD>"])

    optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=0.9,step_size=5)

    input_sentence ="hi there"



    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, train_dataloader, scheduler)
    for epoch in range(10):
        train_ga_hf(model, dataloader, criterion,optimizer,scheduler, epoch,vocab_data,accelerator, DEVICE)
        scheduler.step()
        evaluate(model,val_dataloader,criterion,DEVICE)



    while True:
        input_sentence = input("say something: ")
        print("greedy: ",generate_output(model, input_sentence, vocab_data, max_length=50))
        print("topk: ",generate_output_top_k(model, input_sentence, vocab_data, max_length=50, top_k=10))
    # Training loop (Consider writing a function for this/two separate functions for training and validation)

    # Evaluation by feeding the model with one input sentence at a time

    pass


