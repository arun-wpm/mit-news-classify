import sys

import csv
import itertools as it
import numpy as np
np.random.seed(0)
import pandas as pd


from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply,concatenate,ReLU
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
#from keras.optimizers import Adam # the Keras version of Adam causes a error
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt

import numpy as np
from keras import utils
from keras.utils import to_categorical

import sklearn
import nltk

from keras.preprocessing.text import one_hot

# tokenization


class Tokenizer:
    def __init__(self):
        import re
        import nltk
        self.word_to_token = {}
        self.token_to_word = {}

        self.word_to_token['<unknown>'] = 0
        self.token_to_word[0] = '<unknown>'

        self.vocabulary=[]
        self.vocabulary.append('<unknown>')

    def get_word_to_token(self):
        return self.word_to_token

    def get_token_to_word(self):
        return self.token_to_word

    def build_tokenizer(self, corpus, Cutoff=30):

        # only keep word above certain frequency
        WordCount = {}
        for document in corpus:
            document = document.strip().lower()

            all_words = nltk.word_tokenize(document)

            for word in all_words:
                if word not in WordCount:
                    WordCount[word]=1
                else:
                    WordCount[word]+=1

        for Key , Value in WordCount.items():

            if Value >= Cutoff:
                self.vocabulary.append(Key)
                self.word_to_token[Key] = len(self.vocabulary)-1
                self.token_to_word[len(self.vocabulary)-1] = Key

        print("tokenizer with vocab size of "+str(len(self.vocabulary)))

    def tokenize(self, corpus):
        tokenized= []
        for document in corpus:
            document = document.strip().lower()
            all_words = nltk.word_tokenize(document)
            document_tokens=[]
            for word in all_words:
                if word not in self.word_to_token:
                    document_tokens.append(0)
                else:
                    document_tokens.append(self.word_to_token[word])
            tokenized.append(document_tokens)
        return tokenized


#
#
# def Categorize(Candiates,AllItems):
#     IndexID=[]
#     Categories=[]
#     for x in Candiates:
#             if x not in AllItems:
#                 IndexID.append(len(AllCategories))
#                 Categories.append("Unknown")
#             else:
#                 for i in range(len(AllItems)):
#                     if x==AllItems[i]:
#                         IndexID.append(i)
#                         Categories.append(AllItems[i])
#
#     return(np.asarray(IndexID),np.asarray(Categories))

#train_labels,_ =Categorize(train_labels ,AllCategories)
#val_labels,_=Categorize(val_labels ,AllCategories)
#test_labels ,_=Categorize(test_labels  ,AllCategories)

import sys

import sys


#prepare context and target data

def create_cbow_dataset(text, context_size):
    data = []
    for i in range(context_size, len(text) - context_size):
        context = [text[j] for j in range(i-context_size, i)]+[text[j] for j in range(i+1, i+context_size+1)]

        target = text[i]
        data.append((context, target))
    return data


class Word2VecModel():
    def __init__(self, vocab_size, embed_dim, hidden_size, context_size):
        import nltk
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.context_size = context_size
        self.hidden_size = hidden_size

    def build_model(self,):
        Inputs = Input(shape=(self.context_size * 2,))
        EmbeddingL = Embedding(input_dim=self.vocab_size, output_dim=self.embed_dim, input_length=self.context_size * 2)(Inputs)
        Hidden = Flatten()(EmbeddingL)
        Hidden = Dense(self.hidden_size, activation='relu')(Hidden)
        Hidden = Dense(self.hidden_size, activation='relu')(Hidden)
        Output = Dense(self.vocab_size,activation='softmax')(Hidden)

        self.Model_Pre = Model(inputs=Inputs, outputs=Output)
        self.Model_Pre.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        self.Model_embed = Model(inputs=Inputs, outputs=EmbeddingL)

    def TrainModel(self, X, Y, n_batch, n_epochs):
        self.Model_Pre.fit(X, Y, batch_size=n_batch, epochs=n_epochs)


def get_ngrams(tokenized_corpus, window_size):
    ngrams = []
    for i, review in enumerate(tokenized_corpus):
        for j, word in enumerate(review):
            min_ind = max(0, j-window_size)
            max_ind = min(len(review), j+window_size+1)
            ctx = np.zeros(2 * window_size, dtype=np.int64)
            for ik, k in enumerate(range(min_ind, j)):
                ctx[ik] = review[k]
            for ik, k in enumerate(range(j+1, max_ind)):
                ctx[window_size+ik] = review[k]
            ngrams.append((ctx, review[j]))
    return ngrams


def learn_reps_word2vec(ngrams, vocab_size, embed_dim, hidden_size, context_size, n_epochs, n_batch):

    # convert to Numpy arrage of X and Y
    X = []
    Y = []
    for i in range(len(ngrams)):
        X.append(ngrams[i][0])
        Y.append(ngrams[i][1])
    X = np.stack(X)
    Y = np.stack(Y)

    Y = to_categorical(Y,num_classes=vocab_size)

    # build model
    MODEL = Word2VecModel(vocab_size, embed_dim,hidden_size,context_size)
    MODEL.build_model()
    print(MODEL.Model_Pre.summary())

    # train model
    MODEL.TrainModel(X, Y, n_batch, n_epochs)

    # get matrix of word embeding
    Word_embeding = MODEL.Model_embed.get_weights()[0]
    # Word_embeding=MODEL.Model_embed.predict(X)
    # Word_embeding=np.mean(Word_embeding,1)#mean of the output embedding

    return Word_embeding


# vectorize and aggregate

def vectorize_aggregate(TokenizedData,reps_word2vec):
    # use mean emebding across of words for classfication
    Aggregated = []
    for Tokens in TokenizedData:
        if len(Tokens) == 0:
            Tokens = np.asarray([0])
        Aggregated.append(np.mean(reps_word2vec[np.unique(Tokens),:],axis=0))

    Aggregated = np.vstack(Aggregated)

    return Aggregated


def Categorize(Candiates, AllItems):
    IndexID = []
    Categories = []
    for x in Candiates:
        if x not in AllItems:
            IndexID.append(len(AllItems))
            Categories.append("Unknown")
        else:
            for i in range(len(AllItems)):
                if x==AllItems[i]:
                    IndexID.append(i)
                    Categories.append(AllItems[i])

    return np.asarray(IndexID), np.asarray(Categories)

# Classification model


def train_model(xs, ys, n_batch=500, n_epochs=24):

    model = Sequential([
    Dense(128, input_shape=(xs.shape[1],)),
    Activation('relu'),
    Dense(64),
    Activation('relu'),
    Dense(32),
    Activation('relu'),
    Dense(ys.shape[1]),
    Activation('sigmoid'),
    ])

    model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    model.fit(xs, ys,batch_size=n_batch, epochs=n_epochs)
    # model.predict(FeaturizedX)
    return model


def eval_model(model, xs, ys):
    pred_ys = model.predict(xs)
    pred_ys[np.isnan(pred_ys)] = 0
    ACC = np.sum(np.argmax(pred_ys, 1) == np.argmax(ys == 1, 1))/ys.shape[0]
    print("test accuracy", ACC)
    return ACC
