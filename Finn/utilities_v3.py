# Utility functions for topic classification functions
# Original code by Dianbo, modified and added to by Finn

import re
import csv
import math
import nltk
import numpy as np

from keras.layers import Embedding, Lambda, Dense
from keras.models import Sequential
from keras import backend as K

np.random.seed(0)

# tokenization


class Tokenizer:
    def __init__(self):
        self.word_to_token = {}
        self.token_to_word = {}

        self.word_count = {}

        self.word_to_token['<unknown>'] = 0
        self.token_to_word[0] = '<unknown>'

        self.vocabulary = []
        self.vocabulary.append('<unknown>')
        self.vocab_size = 1

        self.min_occur = 30

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

    def fit(self, corpus):
        for review in corpus:
            review = review.strip().lower()
            words = re.findall(r"[\w']+|[.,!?;]", review)
            for word in words:
                if word not in self.word_count:
                    self.word_count[word] = 0
                self.word_count[word] += 1

        for review in corpus:
            review = review.strip().lower()
            words = re.findall(r"[\w']+|[.,!?;]", review)
            for word in words:
                if self.word_count[word] < self.min_occur:
                    continue
                if word in self.word_to_token:
                    continue
                self.word_to_token[word] = self.vocab_size
                self.token_to_word[self.vocab_size] = word
                self.vocab_size += 1

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


def tokenize(corpus, tokenizer):
    tokenized = []
    for document in corpus:
        document = document.strip().lower()
        all_words = re.findall(r"[\w']+|[.,!?;]", document)
        # all_words = nltk.word_tokenize(document)
        document_tokens = []
        for word in all_words:
            if word not in tokenizer:
                document_tokens.append(0)
            else:
                document_tokens.append(tokenizer[word])
        tokenized.append(document_tokens)
    return tokenized


def loadcsv(filename):
    with open(filename, newline='') as f:
        return list(csv.reader(f))


def get_topic(number):
    if not math.isnan(number):
        return int(number)


def create_one_hot(item, num_items):
    one_hot = np.zeros(num_items)
    for thing in item:
        one_hot[int(thing)] += 1
    return one_hot


def get_mapping(topicfile):
    data = loadcsv(topicfile)
    id_mapping = {int(t[1]): int(t[0]) for t in data[::2][1:]}
    topic_mapping = {int(t[0]): t[2] for t in data[::2][1:]}
    return id_mapping, topic_mapping


def long_to_short_topic_ids(topics, id_mapping):
    new_topics = []
    for topic in topics:
        if topic in id_mapping:
            new_topics.append(id_mapping[topic])
    return new_topics


# Classification model

def train_model(xs, ys, vocab_size, max_length, n_batch=500, n_epochs=24):

    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=max_length))
    model.add(Lambda(lambda x: K.sum(x, axis=1), input_shape=(max_length, vocab_size)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(ys.shape[1], activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(xs, ys, batch_size=n_batch, epochs=n_epochs)
    return model


def eval_model(model, xs, ys):
    pred_ys = model.predict(xs)
    pred_ys[np.isnan(pred_ys)] = 0
    ACC = np.sum(np.argmax(pred_ys, 1) == np.argmax(ys == 1, 1))/ys.shape[0]
    print("test accuracy", ACC)
    return ACC
