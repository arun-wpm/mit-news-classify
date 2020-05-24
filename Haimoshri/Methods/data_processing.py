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
    vocabulary = dict()
    index=0
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

    
