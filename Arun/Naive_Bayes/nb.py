"""
Created on Monday May 25 2020 23:27 +0700

@author: arunwpm
"""
import traceback
import numpy
import sys
sys.path.append("/home/euler/miniconda3/lib/python3.7/site-packages")

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from tf_idf import transform_to_tfidf
import pandas as pd
import data_processing as dp
import pickle
from nltk.tokenize import RegexpTokenizer

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

    f.write("Transforming Labels...\n")
    return label_matrix, id_to_label

def evaluate(true, pred):
    # not sure which way we should evaluate, but for now I'll use false positive and false negatives
    f.write("Debugging: " + str(len(true[0])) + ' ' + str(len(pred[0])) + '\n')
    cnt = [[0, 0], [0, 0]]
    for i in range(len(true)):
        for j in range(len(true[i])):
            cnt[true[i][j]][pred[i][j]] += 1
    
    f.write(str(cnt[0][0]) + ' ' + str(cnt[0][1]) + ' ' + str(cnt[1][0]) + ' ' + str(cnt[1][1]) + '\n')
    f.write("False Positive Rate " + str(cnt[0][1]/(cnt[0][1] + cnt[0][0])) + '\n')
    f.write("False Negative Rate " + str(cnt[1][0]/(cnt[1][0] + cnt[1][1])) + '\n')

def score(true, pred):
    #using the same scoring method as in jiannatags.py
    correct = 0
    falseneg = 0
    falsepos = 0
    for i in range(len(true)):
        for j in range(len(true[i])):
            if true[i][j] == pred[i][j]:
                correct += 1
            elif true[i][j] > pred[i][j]:
                falseneg += 1
            else:
                falsepos += 1
    
    f.write("Correct Rate " + str(correct/len(true)) + '\n')
    f.write("False Positive Rate " + str(falseneg/len(true)) + '\n')
    f.write("False Negative Rate " + str(falsepos/len(true)) + '\n')

def preprocess(document):
    document = document.lower()
    tokenizer = RegexpTokenizer('\w+')
    return tokenizer.tokenize(document)

def get_first_label(labels):
    for i in range(len(labels)):
        labels[i] = -1 if len(labels[i]) == 0 else labels[i][0]
    return labels

if __name__ == "__main__":
    f = open("results.txt", 'a+')
    try:
        # the whole corpus
        with open("../../NYTcorpus.p", "rb") as corpus:
            all_data = pickle.load(corpus)
            f.write("Loaded NYTcorpus!\n")
            data = {'Text': []}
            labels = []
            for row in all_data:
                data['Text'].append(row[2])
                labels.append(row[3:])
            
            # labels, labels_dict = transform_labels(labels)
            labels = get_first_label(labels)

            f.write("Labels size: " + str(len(labels)) + ' ' + str(len(labels[0])) + '\n')
            f.write("Labels dictionary:" + str(labels_dict) + '\n')
            for row in labels:
                assert(len(row) == len(labels[-1]))
            
            f.write("Labels have been transformed\n")

        # small dataset by Haimoshri
        # data = pd.read_csv('small_data_h.csv')
        # data = data.fillna(0)

        # preprocess all the data
        # for i in range(len(data['Text'])):
            # data['Text'][i] = preprocess(data['Text'][i])
        # f.write("Data has been tokenized - no lemmatization at the moment")

        data_train, data_test, labels_train, labels_test = train_test_split(data['Text'], labels, test_size=0.15)
        f.write("Data has been split into 85% train, 15% test\n")

        # transform data by tf_idf : tokenize each sample
        count_vect = CountVectorizer()
        data_train_counts = count_vect.fit_transform(data_train)
        f.write("Words are tokenized, vocabulary built from training data\n")
        tfmer = TfidfTransformer()
        data_train = tfmer.fit_transform(data_train_counts)

        data_test_counts = count_vect.transform(data_test)
        data_test = tfmer.transform(data_test_counts)
        f.write("TF-IDF done:" + str(data_train.shape) + ' ' + str(data_test.shape) + '\n')

        # vocab = dp.build_vocabulary({'Text': data_train}) #should use tokenizer rather than re split?
        # tf_idf = transform_to_tfidf({'Text': data_train}, vocab)

        ovrnb = OneVsRestClassifier(MultinomialNB()).fit(data_train, labels_train)
        f.write("One vs Rest Classfier, Multinomial Naive Bayes fitted data_train\n")
        labels_test_pred = ovrnb.predict(data_test)
        f.write("One vs Rest Classfier, Multinomial Naive Bayes predicts labels for data_train\n")

        # evaluate(labels_test, labels_test_pred)
        score(labels_test, labels_test_pred)
    except Exception:
        traceback.print_exc(file=f)

    f.close()
