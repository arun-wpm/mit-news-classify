"""
Created Tuesday October 6 2020 22:14 +0700

@author: arunwpm
"""
import os

from mitnewsclassify import download

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
from torch.cuda import empty_cache, memory_summary

import traceback
import csv
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def loadcsv(filename):
    with open(filename, newline='') as f: 
        return list(csv.reader(f))

model = None
ld = None
id2tag = {}

def initialize( modelfile="model_pentasemble.h5", 
                ldloc = 'labelsdict.p', #name of the labels dictionary generated by nb.py (should be labels_dict.csv)
                id2tagloc = 'nyt-theme-tags.csv' #name of the conversion table from tag id to tag name for NYTcorpus
                ):
    global model
    global ld
    global id2tag

    # warning
    print("WARNING This model will consume a lot of memory, which can render your computer unusable. Please make sure that you have sufficient memory!")

    # get package directory
    pwd = os.path.dirname(os.path.abspath(__file__))
    pwd += "/data/pentasemble/"
    if (not os.path.isdir(pwd)):
        answer = input("The model files have not been downloaded and the methods will not work. Would you like to download them? [y/n] ")
        if answer == 'y':
            download.download('pentasemble')

    print("Initializing...")
    # initialize the trained model
    model = load_model(pwd + modelfile)
    print("Model...")
    
    # initialize the matrix index -> tag id file and the tag id -> tag name file
    with open(pwd + ldloc, "rb") as ldin:
        ld = pickle.load(ldin)
    # ld = loadcsv(pwd + ldloc)
    id2tag_table = loadcsv(pwd + id2tagloc)
    for row in id2tag_table:
        if row == []:
            continue
        id2tag[row[1]] = row[2]
    print("Miscellaneous...")

def gettags(txt):
    from mitnewsclassify import gpt2
    vec0r = gpt2.getfeatures(txt)
    # print(memory_summary())

    from mitnewsclassify import distilbert
    vec0r = np.concatenate((vec0r, distilbert.getfeatures(txt)), axis=1)
    # print(memory_summary())

    from mitnewsclassify import doc2vec
    vec0m = doc2vec.getfeatures(txt)
    # print(memory_summary())

    from mitnewsclassify import tfidf, tfidf_bi
    vec0l = tfidf.getfeatures(txt)
    vec0l = np.concatenate((vec0l, tfidf_bi.getfeatures(txt)), axis=1)
    # print(memory_summary())

    vec0 = np.concatenate((vec0l, vec0m, vec0r), axis=1) #workaround for tf not being able to free gpu memory

    # print(vec0)

    if (model is None):
        initialize()
    # print(memory_summary())
    mat = model.predict(vec0)
    # print(mat)

    tags = []
    for i in range(len(mat[0])):
        if float(mat[0][i]) >= 0.5:
            tags.append(id2tag[ld[i]])
    # print(tags)

    return tags

if __name__ == "__main__":
    while True:
        txt = input("Enter text: ")
        gettags(txt)