# Code to prdice topics for ImproveTheNews.com
# Written by Finn

import sys

from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import pandas as pd
import numpy as np
import nltk
import csv

max_length = 25000

'''

NOTES ABOUT THIS CODE

When you start running the code, you can pass in either no arguments,
in which case it will use the CNN
model, tokenizer, and embeddings that I have trained. Otherwise you can pass
in arguments for each of these
once we come up with better/different models.

Then to get topic predictions you call the get_topic function passing in a tsv
of the form ["url", "title", "text"].
It will then return a list of lists of topics for each of the articles.

The current threshold is set at 0.3 to give something a topic but can be
easily changed.

'''

topic_threshold = 0.2

if len(sys.argv) == 3:
    model_path = sys.argv[1]
    tok_path = sys.argv[2]

else:
    model_path = 'updated/topic_classifier.h5'
    tok_path = 'updated/tokenizer.csv'

model = load_model(model_path)
tok = pd.read_csv(tok_path, header=None, index_col=0, squeeze=True).to_dict()


def tokenize(corpus, tokenizer):
    tokenized = []
    for document in corpus:
        document = document.strip().lower()
        all_words = nltk.word_tokenize(document)
        document_tokens = []
        for word in all_words:
            if word not in tokenizer:
                document_tokens.append(0)
            else:
                document_tokens.append(tokenizer[word])
        tokenized.append(document_tokens)
    return tokenized


def vectorize_aggregate(tokenized_data, reps_word2vec):
    aggregated = []
    for tokens in tokenized_data:
        if len(tokens) == 0:
            tokens = np.asarray([0])
        aggregated.append(np.mean(reps_word2vec[np.unique(tokens), :], axis=0))

    aggregated = np.vstack(aggregated)

    return aggregated


def probabilities_to_onehot(probabilities):

    probabilities[np.isnan(probabilities)] = 0

    predicted_topics = []
    for prob in probabilities:
        all_topics = []
        topics = np.argwhere(prob > topic_threshold)
        for topic in topics:
            all_topics.append(topic[0])
        predicted_topics.append(all_topics)
    return predicted_topics


def get_topics(text_path):

    data = loadtsv(text_path)
    texts = [d[2] for d in data]

    tokenized_texts = tokenize(texts, tok)
    input = pad_sequences(tokenized_texts, max_length)

    probabilities = model.predict(input)

    predicted_topics = probabilities_to_onehot(probabilities)

    return probabilities, predicted_topics


def loadtsv(filename):
    with open(filename, newline='') as f:
        return list(csv.reader(f, delimiter="\t"))


def savecsv(filename, list):
    with open(filename, "w", newline="") as f:
        csv.writer(f).writerows(list)


'''
text_path = "tagtraining2019nytimes.tsv"
tags = get_topics(text_path)
savecsv("finn_tags.csv", tags)
print("All done - results in finn_tags.csv")

# Test case to see it in action!
# print(get_topics('test.tsv'))

'''
