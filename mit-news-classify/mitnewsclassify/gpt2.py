"""
Created Wednesday August 19 2020 18:56 +0700

@author: arunwpm, haimoshri, jamiefu
"""
import os

from mitnewsclassify import download
from mitnewsclassify.gpt_model import GPTModel

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import tokenize

import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2Model

import traceback
import csv
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device)

def loadcsv(filename):
    with open(filename, newline='') as f: 
        return list(csv.reader(f))

model = None
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = "[PAD]"
gptmodel = GPT2Model.from_pretrained('gpt2').to(device)
ld = None
id2tag = {}

def initialize( modelfile="gpt_0.5.pth", #Jamie's model
                ldloc = 'labels_dict_gpt.csv', #name of the labels dictionary
                id2tagloc = 'nyt-theme-tags.csv' #name of the conversion table from tag id to tag name for NYTcorpus
                ):
    global model
    global doc2vec
    global ld
    global id2tag

    # warning
    print("WARNING This model will consume a lot of memory, which can render your computer unusable. Please make sure that you have sufficient memory!")

    # get package directory
    pwd = os.path.dirname(os.path.abspath(__file__))
    pwd = pwd + "/data/gpt2/"
    if (not os.path.isdir(pwd)):
        answer = input("The model files have not been downloaded and the methods will not work. Would you like to download them? [y/n] ")
        if answer == 'y':
            download.download('gpt2')

    print("Initializing...")

    # initialize the trained model
    model = GPTModel(768, 538)
    model.load_state_dict(torch.load(pwd + modelfile)['state_dict'])
    with torch.no_grad():
        model.eval()
    print("Model...")
    
    # initialize the matrix index -> tag id file and the tag id -> tag name file
    # with open(pwd + ldloc, "rb") as ldin:
        # ld = pickle.load(ldin)
    ld = loadcsv(pwd + ldloc)
    ld = {int(row[0]):row[1] for row in ld}
    id2tag_table = loadcsv(pwd + id2tagloc)
    for row in id2tag_table:
        if row == []:
            continue
        id2tag[row[1]] = row[2]
    print("Miscellaneous...")

def getfeatures(txt):
    if (model is None):
        initialize()
    encoded_dict = tokenizer([txt], add_special_tokens=True, padding="max_length", truncation=True, max_length=1024, return_tensors="pt", return_attention_mask=True)
    result = encoded_dict['input_ids']
    attention_mask = encoded_dict['attention_mask']
    output = gptmodel(input_ids=encoded_dict['input_ids'].to(device), attention_mask=encoded_dict['attention_mask'].to(device))
    output = output[0][:,-1,:].detach()
    vec1 = output.cpu()
    # print(vec1)

    return vec1

def gettags(txt):
    vec1 = getfeatures(txt)
    # print(vec1)

    with torch.no_grad():
        logits = model(vec1)
        mat = model.act(logits)
    # print(mat)

    tags = []
    for i in range(mat.size()[1]):
        if float(mat[0,i]) >= 0.5:
            tags.append(id2tag[ld[i]])
    # print(tags)

    return tags

if __name__ == "__main__":
    while True:
        txt = input("Enter text: ")
        gettags(txt)