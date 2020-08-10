import sys
#
import csv
import itertools as it
import numpy as np

import pandas as pd

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding
from keras.layers import concatenate, ReLU, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
# from keras.optimizers import Adam # the Keras version of Adam causes a error
# from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt

import numpy as np
from keras import utils
from keras.utils import to_categorical

import math

from utilities_v2 import Tokenizer, train_model, eval_model

from keras.preprocessing.sequence import pad_sequences

np.random.seed(0)


def get_topic(number):
    if not math.isnan(number):
        return int(number)


def load_data(file_name):
    data = pd.read_csv(file_name, sep='\t', names=["url", "title", "text", "topic1", "topic2", "topic3", "topic4",
                                                   "topic5", "topic6", "topic7", "topic8", "topic9", "topic10"])
    all_texts1 = list(data['text'])
    all_topics = []
    all_texts = []
    for i in range(len(all_texts1)):
        if len(str(all_texts1[i])) > 10:
            article_topics = []
            for j in topic_numbers:
                try:
                    article_topics.append(int(data[j][i]))
                except:
                    pass
            all_texts.append(str(all_texts1[i]))
            all_topics.append(article_topics)
    return all_texts, all_topics


def create_one_hot(item, num_items):
    one_hot = np.zeros(num_items)
    for thing in item:
        one_hot[int(thing)] += 1
    return one_hot


if __name__ == "__main__":

    # Code to load in TOPIC_TRAINING_DATA corpus
    # -----------------------------------------------------------------------------------------------------
    print("Starting to load in data")
    topic_numbers = ["topic1", "topic2", "topic3", "topic4", "topic5", "topic6", "topic7", "topic8", "topic9",
                     "topic10"]

    years = ['2019']  # , '2018']
    news_sources = ['cnn']  # ['atlantic', 'blaze', 'breitbart', 'businessinsider', 'buzzfeed', 'cbs', 'cnbc', 'cnn',
    # 'dailycaller', 'dailynews', 'foxnews', 'guardian', 'huffpo', 'latimes', 'nbc', 'newmax',
    # 'newyorker', 'politico', 'reuters', 'slate', 'time', 'usatoday', 'vox', 'wapo',
    # 'wsj', 'wsjblogs', 'yahoonews']
    texts = []
    topics = []

    for year in years:
        for source in news_sources:
            new_texts, new_topics = load_data('tagtraining' + year + source + '.tsv')
            # new_texts, new_topics = load_data('~/DATA1/DIANBO/TOPIC_TRAINING_DATA/tagtraining' + year + source + '.tsv')
            texts += new_texts
            topics += new_topics

    texts = np.array(texts)
    topics = np.array(topics)

    # Shuffle the data
    permutation = np.random.permutation(len(texts))
    texts = texts[permutation]
    topics = topics[permutation]

    num_examples = len(texts)
    val_index = (7 * num_examples) // 8

    # val_index = 100  # DELETE AFTER TESTING
    # num_examples = 150  # DELETE AFTER TESTING

    train_texts = texts[0:val_index]
    train_labels = topics[0:val_index]

    test_texts = texts[val_index:num_examples]
    test_labels = topics[val_index:num_examples]

    num_topics = 600
    print("Loaded in data")
    # -----------------------------------------------------------------------------------------------------

    corpus = train_texts

    embed_dim = 500
    context_size = 5
    hidden_size = 64
    learning_rate = 0.001
    n_epochs = 3
    n_batch = 500

    max_length = 25000

    # step1: train word2vec embedding

    print("Tokenizing Data")

    tokenizer = Tokenizer()
    tokenizer.build_tokenizer(corpus)

    vocab_size = len(tokenizer.vocabulary)

    tokenized_corpus = tokenizer.tokenize(corpus)

    word_to_token = tokenizer.get_word_to_token()
    word_to_token.pop(',', None)
    w = csv.writer(open("updated/tokenizer.csv", "w"))
    for key, val in word_to_token.items():
        w.writerow([key, val])

    # step 2 train topic classification model

    Tokenized_X_train = tokenized_corpus
    # one_hot_x_train = []
    # for tokenized_text in Tokenized_X_train:
    #     one_hot_x_train.append(create_one_hot(tokenized_text, vocab_size))
    # Tokenized_X_train = vectorize_aggregate(Tokenized_X_train, reps_word2vec)
    lengths = [len(i) for i in Tokenized_X_train]
    # max_length = max(lengths)
    # print(max_length)
    Tokenized_X_train = pad_sequences(Tokenized_X_train, maxlen=max_length, padding='post')

    Y_train = []
    for point in train_labels:
        Y_train.append(create_one_hot(point, num_topics))
    Y_train = np.array(Y_train)

    Tokenized_X_test = tokenizer.tokenize(test_texts)
    # one_hot_x_test = []
    # for tokenized_text in Tokenized_X_test:
    #     one_hot_x_test.append(create_one_hot(tokenized_text, vocab_size))
    # Tokenized_X_test = vectorize_aggregate(Tokenized_X_test, reps_word2vec)
    Tokenized_X_test = pad_sequences(Tokenized_X_test, maxlen=max_length, padding='post')

    Y_test = []
    for point in test_labels:
        Y_test.append(create_one_hot(point, num_topics))
    Y_test = np.array(Y_test)

    name = "w2v"

    print(f"{name} features, {Tokenized_X_train.shape[0]} examples")

    print("Starting classifier training")
    model = train_model(Tokenized_X_train, Y_train, vocab_size, max_length, n_batch=250, n_epochs=6)

    model.save("updated/topic_classifier.h5")

    ACC = eval_model(model, Tokenized_X_test, Y_test)
