"""
Created Wednesday August 19 2020 17:46 +0700

@author: arunwpm
"""
import os

from mitnewsclassify import download, utils

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import tokenize

import traceback
import pickle

model = None
ld = None
doc2vec_model = None
id2tag = {}

def initialize( modelfile="model_1200_800_40.h5", 
                doc2vecfile="doc2vec_model",
                ldloc = 'labelsdict.p', #name of the labels dictionary
                id2tagloc = 'nyt-theme-tags.csv', #name of the conversion table from tag id to tag name for NYTcorpus
                google=False # whether to use the model trained on our google dataset or the original nyt annotated corpus dataset
                ):
    global model
    global doc2vec_model
    global ld
    global id2tag

    # warning
    print("WARNING This model will consume a lot of memory, which can render your computer unusable. Please make sure that you have sufficient memory!")

    # get package directory
    pwd = os.path.dirname(os.path.abspath(__file__))
    pwd += "/data/doc2vec/" if google == False else "/gdata/doc2vec/"
    if (not os.path.isdir(pwd)):
        answer = input("The model files have not been downloaded and the methods will not work. Would you like to download them? [y/n] ")
        if answer == 'y':
            download.download('doc2vec', google)

    print("Initializing...")
    # initialize the trained model
    model = load_model(pwd + modelfile)
    print("Model...")

    # initialize the trained doc2vec model
    doc2vec_model = Doc2Vec.load(pwd + doc2vecfile)
    print("Doc2Vec Model...")
    
    # initialize the matrix index -> tag id file and the tag id -> tag name file
    with open(pwd + ldloc, "rb") as ldin:
        ld = pickle.load(ldin)
    # ld = utils.loadcsv(pwd + ldloc)
    id2tag_table = utils.loadcsv(pwd + id2tagloc)
    for row in id2tag_table:
        if row == []:
            continue
        id2tag[row[1]] = row[2]
    print("Miscellaneous...")

def gettags(txt, google=False):
    if (model is None):
        initialize(google=google)
    vec1 = doc2vec_model.infer_vector(list(tokenize(txt)))
    # print(vec1)

    mat = model.predict(np.array([vec1]))
    # print(mat)

    tags = []
    for i in range(len(mat[0])):
        if float(mat[0][i]) >= 0.5:
            tags.append(id2tag[ld[i]])
    # print(tags)

    return tags

def getfeatures(txt, google=False):
    if (model is None):
        initialize(google=google)
    vec1 = doc2vec_model.infer_vector(list(tokenize(txt)))
    # print(vec1)

    extractor = keras.Model(inputs=model.input, outputs=model.get_layer('last_hidden').output)
    features = extractor(np.array([vec1]))
    # print(mat)

    return features

def free():
    global doc2vec_model
    del doc2vec_model

if __name__ == "__main__":
    while True:
        txt = input("Enter text: ")
        gettags(txt)