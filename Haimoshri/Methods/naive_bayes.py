#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 21:49:37 2020

@author: haimoshri

"""

import pandas as pd
import csv
import read_tags as rt
import data_processing as dp

from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier

with open('../Data/small_data_test.csv', 'r') as f:
    file = pd.read_csv(f)
    head_tag = []
    for i in range(593):
        head_tag.append(str(i))
    
    num_tags = rt.num_tags
    
    vocabulary = dp.build_vocabulary(file)
    
    td_matrix = dp.build_bow(vocabulary, file)
    
    labels = dp.build_labels(num_tags, file)
    
    model = OneVsRestClassifier(MultinomialNB())

    model.fit(td_matrix, labels)