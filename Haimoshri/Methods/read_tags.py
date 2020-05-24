#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 17:06:56 2020

@author: haimoshri
"""

import pandas as pd
import csv

f = pd.read_csv('../Data/tags.csv')
f_result = f[['tags_id', 'tag']]

tags_reverse = pd.Series(f_result.tags_id.values,index=f_result.tag).to_dict()

tags = pd.Series(f_result.tag.values,index=f_result.tags_id).to_dict()

tag_index = dict()

count = 0
for i in tags:
    tag_index[i] = count
    count+=1

num_tags = len(tag_index)
