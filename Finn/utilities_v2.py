import sys

import csv
import itertools as it
import numpy as np

import pandas as pd


from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, concatenate,ReLU
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
# from keras.optimizers import Adam # the Keras version of Adam causes a error
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt

import numpy as np
from keras import utils
from keras.utils import to_categorical

import sklearn
import nltk

from keras.preprocessing.text import one_hot
from keras import backend as K

np.random.seed(0)

# tokenization


class Tokenizer:
    def __init__(self):
        import re
        import nltk
        self.word_to_token = {}
        self.token_to_word = {}

        self.word_to_token['<unknown>'] = 0
        self.token_to_word[0] = '<unknown>'

        self.vocabulary = []
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
                    WordCount[word] = 1
                else:
                    WordCount[word] += 1

        for Key, Value in WordCount.items():

            if Value >= Cutoff:
                self.vocabulary.append(Key)
                self.word_to_token[Key] = len(self.vocabulary)-1
                self.token_to_word[len(self.vocabulary)-1] = Key

        print("tokenizer with vocab size of "+str(len(self.vocabulary)))

    def tokenize(self, corpus):
        tokenized = []
        for document in corpus:
            document = document.strip().lower()
            all_words = nltk.word_tokenize(document)
            document_tokens = []
            for word in all_words:
                if word not in self.word_to_token:
                    document_tokens.append(0)
                else:
                    document_tokens.append(self.word_to_token[word])
            tokenized.append(document_tokens)
        return tokenized


# vectorize and aggregate

def vectorize_aggregate(TokenizedData, reps_word2vec):
    # use mean emebding across of words for classfication
    Aggregated = []
    for Tokens in TokenizedData:
        if len(Tokens) == 0:
            Tokens = np.asarray([0])
        Aggregated.append(np.mean(reps_word2vec[np.unique(Tokens), :], axis=0))

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
                if x == AllItems[i]:
                    IndexID.append(i)
                    Categories.append(AllItems[i])

    return np.asarray(IndexID), np.asarray(Categories)

# Classification model


def train_model(xs, ys, vocab_size, max_length, n_batch=500, n_epochs=24):

    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=max_length))
    model.add(Lambda(lambda x: K.sum(x, axis=1), input_shape=(max_length, vocab_size)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(ys.shape[1], activation='sigmoid'))

    # model = Sequential([
    #     Embedding(vocab_size, 100, input_length=max_length),
    #     Flatten(),
    #     Dense(128),
    #     Activation('relu'),
    #     Dense(64),
    #     Activation('relu'),
    #     Dense(32),
    #     Activation('relu'),
    #     Dense(ys.shape[1]),
    #     Activation('sigmoid'),
    # ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(xs, ys, batch_size=n_batch, epochs=n_epochs)
    return model


def eval_model(model, xs, ys):
    pred_ys = model.predict(xs)
    pred_ys[np.isnan(pred_ys)] = 0
    ACC = np.sum(np.argmax(pred_ys, 1) == np.argmax(ys == 1, 1))/ys.shape[0]
    print("test accuracy", ACC)
    return ACC
