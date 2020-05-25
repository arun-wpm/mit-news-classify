"""
Created on Monday May 25 2020 23:27 +0700

@author: arunwpm
"""
import numpy
import sys
sys.path.append("/home/euler/miniconda3/lib/python3.7/site-packages")

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from tf_idf import transform_to_tfidf
import pandas as pd
import data_processing as dp
import pickle

def transform_labels(in_labels):
    # from the input labels, in the format of list of list of labels pertaining to document at row i,
    # output a multilabel matrix where row i column j is 1 if document i contains label j and 0 otherwise
    # probably is somewhat the same as some parts of Haimoshri's small_data_gen.py, didn't see at first oops
    label_to_id = {}
    id_to_label = {}
    cnt = 0
    label_matrix = []
    for row in in_labels:
        label_matrix.append([])
        for label in row:
            if label not in label_to_id:
                label_to_id[label] = cnt
                id_to_label[cnt] = label
                cnt += 1
            i = label_to_id[label]
            if i >= len(label_matrix[-1]):
                label_matrix[-1].extend([0]*(i + 1 - len(label_matrix[-1])))
            label_matrix[-1][i] = 1
    
    for i in range(len(label_matrix)):
        if len(label_matrix[i]) < len(label_to_id):
            label_matrix[i].extend([0]*(len(label_to_id) - len(label_matrix[i])))

    print("Transforming Labels...")
    return label_matrix, id_to_label

def evaluate(pred, true):
    # not sure which way we should evaluate, but for now I'll use false positive and false negatives
    cnt = [[0, 0], [0, 0]]
    for i in range(len(true)):
        for j in range(len(true[i])):
            cnt[true[i][j]][pred[i][j]] += 1
    print("False Positive Rate ", cnt[0][1]/(cnt[0][1] + cnt[0][0]))
    print("False Negative Rate ", cnt[1][0]/(cnt[1][0] + cnt[1][1]))

if __name__ == "__main__":
    # the whole corpus
    with open("../../NYTcorpus.p", "rb") as corpus:
        all_data = pickle.load(corpus)
        print("Loaded NYTcorpus!")
        data = {'Text': []}
        labels = []
        for row in all_data:
            data['Text'].append(row[2])
            labels.append(row[3:])
        
        labels, labels_dict = transform_labels(labels)
        print(labels_dict)
        for row in labels:
            assert(len(row) == len(labels[-1]))

    # small dataset by Haimoshri
    # data = pd.read_csv('small_data_h.csv')
    # data = data.fillna(0)

    data_train, data_test, labels_train, labels_test = train_test_split(data['Text'], labels, test_size=0.5)

    # transform data by tf_idf : tokenize each sample
    count_vect = CountVectorizer()
    data_train_counts = count_vect.fit_transform(data_train)
    data_train = TfidfTransformer().fit_transform(data_train_counts)
    data_test = TfidfTransformer().transform(data_train_counts)

    # vocab = dp.build_vocabulary({'Text': data_train}) #should use tokenizer rather than re split?
    # tf_idf = transform_to_tfidf({'Text': data_train}, vocab)

    ovrnb = OneVsRestClassifier(MultinomialNB()).fit(data_train, labels_train)
    labels_test_pred = ovrnb.predict(data_test)

    evaluate(labels_test, labels_test_pred)
