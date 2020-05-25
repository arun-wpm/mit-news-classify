#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 09:46:24 2020

@author: haimoshri
"""

import math
import re
import numpy as np
import csv
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import pandas as pd
import data_processing as dp        

def transform_to_tfidf(data, vocab):
    vectorizer = CountVectorizer(vocabulary=vocab)
    transformer = TfidfTransformer()
    count = vectorizer.fit_transform(data['Text'])
    tf_idf = transformer.fit_transform(count)
    return tf_idf

data = pd.read_csv('small_data_h.csv')

data = data.fillna(0)

vocab = dp.build_vocabulary(data)

tf_idf = transform_to_tfidf(data, vocab)

        