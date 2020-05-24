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
    vocabulary['UNK'] = 0
    index=1
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
    """Function to build term document matrix from Text where data is a Data Frame."""
    
    result = np.zeros((len(data), len(vocab)))
    count = 0
    for i in data['Text']:
        bow_vector = np.zeros(len(vocab))
        one_example = re.split('; |,| |\n| .',i)
        for j in one_example:
            if i in vocab:
                bow_vector[vocab[i]] = 1
            else:
                bow_vector[vocab['UNK']] = 1
        result[count, :] = bow_vector
        count+=1
    
    return result
    