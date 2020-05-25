#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 14:11:57 2020

@author: haimoshri
"""

#import pandas as pd

# with open('small_data.csv',newline='') as f:
#     r = csv.reader(f)
#     data = [line for line in r]
# with open('small_data_h.csv','w',newline='') as f:
#     w = csv.writer(f)
#     head = ['URL', 'Title', 'Text']
#     for i in range(600):
#         head+=[str(i)]
#     w.writerow(head)
#     w.writerows(data)


import pandas as pd
import csv
import read_tags as t

with open('small_data.csv',newline='') as f:
    r = csv.reader(f)
    data = []
    for line in r:
        d = line[0:3]
        c_tags = [0 for i in range(t.num_tags)]
        for i in line[3:]:
            c_tags[t.tag_index[int(i)]] = 1
        line = d+c_tags
        data.append(line)

with open('small_data_test.csv','w',newline='') as f:
    w = csv.writer(f)
    head = ['URL', 'Title', 'Text']
    for i in range(t.num_tags):
        head+=[str(i)]
    w.writerow(head)
    w.writerows(data)
