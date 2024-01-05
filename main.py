'''
Template for the 4th assignment
Student: YASSINE OUESLATI
'''

############################
# Packages
############################
import torch
import torch.nn as nn
import math
import regex as re
import ast
import json
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

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

        for s1 , s2 in tqdm(self.pairs):
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
        tokenized_answer.append(self.word2index["<EOS>"])

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
            tokenized_sentence.append(self.word2index[word])
        return tokenized_sentence

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
        for _ in tqdm(range(len(self.pairs))):
            if self.len_pair(index)>max_length:
                self.pairs.pop(index)

            else:
                index+=1



    def remove_word(self,word):


        removal_indices = set()
        for index, (p1, p2) in enumerate(self.tokenized_pairs):
            if word in p1 or word in p2:
                removal_indices.add(index)

        # Remove all marked pairs in a single operation
        self.pairs = [pair for i, pair in enumerate(self.pairs) if i not in removal_indices]
        #self.tokenized_pairs= [pair for i, pair in enumerate(self.tokenized_pairs) if i not in removal_indices]
    def remove_pairs_word(self,min_freq):

        low_freq_words = {word for word, count in self.word_count.items() if count <= min_freq}

        for word in tqdm(low_freq_words):
            self.remove_word(word)



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
        pass

    def __len__(self):
        # TODO how many pairs do we have?
        pass

    def __getitem__(self, ix):
        # TODO returns two tensors (question, answer) of the pair at index ix
        # TODO the tensors should be of type torch.tensor and should contain integers (word indices)
        pass


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
    def __init__(self, vocab_size, d_model=512, pad_id=0, encoder_layers=6, decoder_layers=6, dim_feedforward=2048,
                 num_heads=8, dropout_p=0.1):
        super().__init__()
        # TODO add an embedding layer
        # TODO add a positional encoding layer
        # TODO add a transformer layer, you can use nn.Transformer. You can use the default values for the parameters, but what about batch_first?
        # TODO add a linear layer. Note: output should be probability distribution over the vocabulary

        # Stuff you may need
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.num_heads = num_heads

    def create_padding_mask(self, x, pad_id=0):
        # TODO create a boolean mask for the <PAD> tokens
        pass

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
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1), dtype=torch.bool).to(
            tgt.device)  # (T, T)
        # Expand to make it N * num_heads, T, T
        tgt_mask = tgt_mask.unsqueeze(0).repeat(tgt.size(0) * self.num_heads, 1, 1)  # (N, T, T)
        # Transformer
        output = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask,
                                  tgt_key_padding_mask=tgt_pad_mask,
                                  memory_key_padding_mask=memory_key_padding_mask)  # (N, T, E)
        # Linear layer
        output = self.linear(output)  # (N, T, V)
        return output


############################
# Methods
############################



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





def save_data(array):

    with open('data.pkl', 'wb') as f:
        pickle.dump(array, f)
    return 0


def load_data():
    with open('data.pkl', 'rb') as f:
        array = pickle.load(f)
    return array


def plot_frequency(array,file_name):
    array.sort()

    plt.hist(array[:40000])
    plt.show()
    """
    print(array)
    df = pd.DataFrame(array, columns=['words count'])
    fig = sns.displot(df, x="words count", discrete=True)
    fig.savefig(fname=file_name)
    """
    return 0


if __name__ == "__main__":
    # !!! Don't change the seed !!!
    torch.manual_seed(42)
    # !!!!!!

    # Download the data
    corpus = Corpus()
    corpus = inspect_files()

    # Create the pairs

    vocab = Vocabulary(name="eng",pairs=create_list_pairs(corpus))

    # Tokenize the data
    vocab = load_data()
    """
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
    """
    #print(vocab.word_count.values())
    plot_frequency(list(vocab.word_count.values()), "word_count.png")
    print(len(vocab.pairs))
    vocab.remove_pairs_word(5)
    print(len(vocab.pairs))
    save_data(vocab)
    # SAVE and put the code above into a function that you will call if you need to generate something slightly different

    # Training loop (Consider writing a function for this/two separate functions for training and validation)

    # Evaluation by feeding the model with one input sentence at a time

    pass

