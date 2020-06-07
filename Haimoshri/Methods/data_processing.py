#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 15:30:51 2020

@author: haimoshri
"""

import numpy as np
import csv

from nltk.corpus import stopwords

import math
import re
import numpy as np
import csv
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import pandas as pd
 

stopword = set(stopwords.words('english'))
stopword.add('i\'m')
stopword.add('can\'t')
stopword.add('ie')
stopword.add('may')
stopword.add('etc')

def build_vocabulary(data):
    """Builds a vobaulary from Text column of data that is a Data Frame."""
    
    vocabulary = dict()
    vocabulary['UNK'] = 1
    vocabulary['PAD'] = 0
    index=2
    for i in data['Text']:
      words = re.split('; |,| |\n| .',i)
      for word in words:
          if word in stopword:
              continue
          try:
              vocabulary[word]
          except:
              vocabulary[word] = index
              index+=1
    return vocabulary

    
def build_bow(vocab, data):
    """Function to build term document matrix, bow style from Text where data is a Data Frame."""
    
    result = np.zeros((len(data), len(vocab)))
    count = 0
    for i in data['Text']:
        bow_vector = np.zeros(len(vocab))
        one_example = re.split('; |,| |\n| .',i)
        for j in one_example:
            if j in vocab:
                bow_vector[vocab[j]] = 1
            else:
                bow_vector[vocab['UNK']] = 1
        result[count, :] = bow_vector
        count+=1
    
    return result

def build_index_vec(vocab, data, max_len):
    """Function to build term document matrix, bow style from Text where data is a Data Frame."""
    
    result = np.zeros((len(data), max_len))
    count = 0
    ind = 0
    for i in data['Text']:
        index_vector = np.zeros(max_len)
        one_example = re.split('; |,| |\n| .',i)
        for j in one_example:
            if j in vocab:
                index_vector[ind] = vocab[j]
            else:
                index_vector[ind] = 1
            ind+=1
            #print (ind)
            if ind >= max_len:
                ind = 0
                break
        result[count, :] = index_vector
        count+=1
    
    return result



def build_labels(num_labels, data):
    """Function to build label matrix, binary valued vector style where data is a Data Frame."""
    
    head_tags = []
    for i in range(num_labels):
        head_tags.append(str(i))
    
    labels = data[head_tags]
    
    return labels.to_numpy()