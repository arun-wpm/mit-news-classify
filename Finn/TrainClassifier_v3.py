# Topic classifier for nyt_corpus
# Initial impementation by Dianbo, adapted by Finn

import csv
import pickle
import numpy as np
import pandas as pd

from utilities_v3 import Tokenizer, train_model
from utilities_v3 import create_one_hot, tokenize
from utilities_v3 import get_mapping, long_to_short_topic_ids

from keras.preprocessing.sequence import pad_sequences

from evaluation import false_eval

np.random.seed(0)


if __name__ == "__main__":

    # Code to load in TOPIC_TRAINING_DATA corpus
    # ---------------------------------------------------------------------
    # f = open("nyt2/tracking.txt", "w")
    # print("Starting to load in data")
    # f.write("Starting to load in data")

    topicfile = "nyt-theme-tags.csv"
    id_mapping, topic_mapping = get_mapping(topicfile)

    infile = "../../nyt_corpus/NYTcorpus.p"
    articles = pickle.load(open(infile, "rb"))

    texts = [a[2] for a in articles[1:]]
    long_topics = [list(map(int, a[3:])) for a in articles[1:]]
    topics = []
    for topic in long_topics:
        topics.append(long_to_short_topic_ids(topic, id_mapping))

    # Shuffle the data
    # texts = np.array(texts)
    # topics = np.array(topics)
    # permutation = np.random.permutation(len(texts))
    # texts = texts[permutation]
    # topics = topics[permutation]

    num_examples = len(texts)
    val_index = (7 * num_examples) // 8

    # val_index = 100  # DELETE AFTER TESTING
    # num_examples = 150  # DELETE AFTER TESTING

    train_texts = texts[0:val_index]
    train_labels = topics[0:val_index]

    test_texts = texts[val_index:]
    test_labels = topics[val_index:]

    num_topics = 594
    print("Loaded in data")
    # f.write("Loaded in data \n")
    # ---------------------------------------------------------------------

    embed_dim = 500
    context_size = 5
    hidden_size = 64
    learning_rate = 0.001
    n_epochs = 3
    n_batch = 500

    max_length = 25000

    # step1: train word2vec embedding

    # f = open("nyt2/started.txt", "w")
    # f.write("Tokenization started")
    # f.close()
    #
    # # print("Tokenizing Data")
    # # f.write("Tokenizing Data \n")
    #
    # tokenizer = Tokenizer()
    # # tokenizer.build_tokenizer(train_texts)
    # tokenizer.fit(train_texts)
    #
    # vocab_size = len(tokenizer.vocabulary)
    #
    # word_to_token = tokenizer.get_word_to_token()
    # word_to_token.pop(',', None)
    # w = csv.writer(open("nyt2/tokenizer.csv", "w"))
    # for key, val in word_to_token.items():
    #     w.writerow([key, val])
    #
    # f = open("nyt2/finished.txt", "w")
    # f.write("Tokenization finished")
    # f.close()

    tokenizer_path = "nyt2/tokenizer.csv"
    word_to_token = pd.read_csv(tokenizer_path, header=None, index_col=0, squeeze=True).to_dict()
    vocab_size = len(word_to_token)

    tokenized_corpus = tokenize(train_texts, word_to_token)
    print("Tokenized train texts")

    # step 2 train topic classification model

    X_train = tokenized_corpus
    # one_hot_x_train = []
    # for tokenized_text in Tokenized_X_train:
    #     one_hot_x_train.append(create_one_hot(tokenized_text, vocab_size))
    # Tokenized_X_train = vectorize_aggregate(Tokenized_X_train, reps_word2vec)
    lengths = [len(i) for i in X_train]
    # max_length = max(lengths)
    # print(max_length)
    print("Starting Padding")
    for i, article in enumerate(X_train):
        if len(article) < max_length:
            padded_article = article + [0]*(max_length - len(article))
        else:
            padded_article = article[:max_length]
        X_train[i] = padded_article
    # X_train = pad_sequences(X_train, maxlen=max_length, padding='post')
    print("Ended Padding")
    Y_train = []
    for point in train_labels:
        Y_train.append(create_one_hot(point, num_topics))
    Y_train = np.array(Y_train)

    print("Starting Test Tokenization")
    X_test = tokenize(test_texts, word_to_token)
    # one_hot_x_test = []
    # for tokenized_text in Tokenized_X_test:
    #     one_hot_x_test.append(create_one_hot(tokenized_text, vocab_size))
    # Tokenized_X_test = vectorize_aggregate(Tokenized_X_test, reps_word2vec)

    print("Starting Padding")
    # TODO: NEED TO TRY TO DO EMBEDDING WITH VARIABLE SIZED INPUT
    for i, article in enumerate(X_test):
        if len(article) < max_length:
            padded_article = article + [0]*(max_length - len(article))
        else:
            padded_article = article[:max_length]
        X_test[i] = padded_article
    print("Ended Padding")
    # X_test = pad_sequences(X_test, maxlen=max_length, padding='post')

    Y_test = []
    for point in test_labels:
        Y_test.append(create_one_hot(point, num_topics))
    Y_test = np.array(Y_test)

    print("Starting classifier training")
    # f.write("Starting classifier training \n")

    model = train_model(X_train, Y_train, vocab_size, max_length, 250, 6)

    model.save("nyt2/topic_classifier.h5")

    total, f_n, f_p = false_eval(Y_test, model.predict(X_test))
    f = open("nyt2/evaluation.txt", "w")
    f.write(str(total) + str(f_n) + str(f_p))
    f.close()

    # print("Total topics to be predicted:", total)
    # print("False negatives:", f_n)
    # print("False positives:", f_p)
