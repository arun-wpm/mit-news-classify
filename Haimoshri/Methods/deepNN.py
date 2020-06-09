#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 00:31:33 2020

@author: haimoshri
"""

import pickle
import pandas as pd
import csv
import read_tags as rt
import data_processing as dp

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from torch.utils.data import DataLoader, TensorDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device)

batch_size = 64
hidden_size = 32
output_size = 593
num_layers = 2
dropout = 0.5
embedding_size = 64
lr = 0.1
epochs = 2 # the dataset is huge
vocab_size = 0
loss_fn = nn.MultiLabelMarginLoss()
max_len = 400

# load csv files
# def load_data(filename):
#     with open(filename, 'r') as f:
#         file = pd.read_csv(f)
#         head_tag = []
#         for i in range(593):
#             head_tag.append(str(i))
        
#         num_tags = rt.num_tags
        
#         vocabulary = dp.build_vocabulary(file)
        
#         vocab_size = len(vocabulary)
        
#         td_matrix = dp.build_index_vec(vocabulary, file, max_len)
        
#         labels = dp.build_labels(num_tags, file)
        
#         return td_matrix, labels, vocab_size

def transform_labels(in_labels):
    # from the input labels, in the format of list of list of labels pertaining to document at row i,
    # output a multilabel matrix where row i column j is 1 if document i contains label j and 0 otherwise
    label_to_id = {}
    id_to_label = {}
    # cnt = 0
    label_matrix = []
    for row in in_labels:
        label_matrix.append([])
        for labelstr in row:
            label = int(labelstr)
            if label not in label_to_id:
                cnt = rt.tag_index[label]
                label_to_id[label] = cnt
                id_to_label[cnt] = label
            i = label_to_id[label]
            if i >= len(label_matrix[-1]):
                label_matrix[-1].extend([0]*(i + 1 - len(label_matrix[-1])))
            label_matrix[-1][i] = 1
    
    for i in range(len(label_matrix)):
        # if len(label_matrix[i]) < len(label_to_id):
        if len(label_matrix[i]) < rt.num_tags:
            label_matrix[i].extend([0]*(rt.num_tags - len(label_matrix[i])))

    print("Transforming Labels...\n")
    return label_matrix, id_to_label

def load_data():
    # Load NYT corpus data
    with open("../../NYTcorpus.p", "rb") as corpus:
        all_data = pickle.load(corpus)
        data = {'Text': []}
        labels = []
        for row in all_data:
            data['Text'].append(row[2])
            labels.append(row[3:])
        
        labels_list, labels_dict = transform_labels(labels)
        print(len(labels_list), len(labels_list[0]))
        labels = np.array(labels_list)
        print(labels.shape)

        # saving the labels dictionary
        # savecsv("labels_dict.csv", list(labels_dict.items()))

        num_tags = rt.num_tags
        vocabulary = dp.build_vocabulary(data)
        vocab_size = len(vocabulary)
        
        td_matrix = dp.build_index_vec(vocabulary, data, max_len)
        
        return td_matrix, labels, vocab_size
        
def prepare_data(td_matrix, labels):
    
        # 80% train; 10% validation; 10% test
        # split_1 = int(4*len(td_matrix)/5)
        # split_2 = int(9*len(td_matrix)/10)

        # 75% train; 10% validation, 15% test
        split_1 = int(len(td_matrix)*0.75)
        split_2 = int(len(td_matrix)*0.85)
        print(td_matrix.shape, labels.shape)
        print(split_1, split_2)

        # np.random.shuffle(td_matrix) # shuffle first as well
        
        train_data_processed = td_matrix[:split_1, :]
        train_labels = labels[:split_1, :]
        
        val_data_processed = td_matrix[split_1:split_2, :]
        val_labels = labels[split_1:split_2, :]
        
        test_data_processed = td_matrix[split_2:, :]
        test_labels = labels[split_2:, :]
        
        train_data_tensor = TensorDataset(torch.from_numpy(train_data_processed), torch.from_numpy(train_labels))
        val_data_tensor = TensorDataset(torch.from_numpy(val_data_processed), torch.from_numpy(val_labels))
        test_data_tensor = TensorDataset(torch.from_numpy(test_data_processed), torch.from_numpy(test_labels))
        
        train_data_loader = DataLoader(train_data_tensor, shuffle=True, batch_size=batch_size)
        val_data_loader = DataLoader(val_data_tensor, shuffle=True, batch_size=batch_size)
        test_data_loader = DataLoader(test_data_tensor, shuffle=True, batch_size=1)
        
        return train_data_loader, val_data_loader, test_data_loader

def savecsv(filename, list):
    with open(filename, "w", newline="") as f:
        csv.writer(f).writerows(list)

# td_matrix, labels, vocab_size = load_data('../Data/small_data_test.csv')
td_matrix, labels, vocab_size = load_data()

train_data_loader, val_data_loader, test_data_loader = prepare_data(td_matrix, labels)

class MyModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, output_size, num_layers, dropout=0):
        super(MyModel, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout, bidirectional=False)
        self.output = nn.Linear(hidden_size, output_size)
        self.activation = nn.Sigmoid()


    def forward(self, x):
        embeds = self.embedding(x)
        outputs1, a = self.lstm(embeds)
        outputs = self.output(outputs1)
        outputs = outputs[:, -1, :]
        outputs = self.activation(outputs)
        return outputs

classifier = MyModel(vocab_size, embedding_size, hidden_size, output_size, num_layers, dropout)
optimizer = optim.SGD(classifier.parameters(), lr=lr, momentum=0.9)
#optimizer = optim.Adam(classifier.parameters(), lr=0.0001)

classifier = classifier.to(device)
classifier.train()

for epoch in range(epochs):

    total_loss = 0
    classifier.train()
    for inputs, labels in train_data_loader:
    
        inputs = inputs.type(torch.LongTensor)

        classifier.zero_grad()
        logits = classifier.forward(inputs)
        logits = logits.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)
        
        loss = loss_fn(logits.squeeze(), labels)
        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    print ('train', total_loss)
    
    total_loss = 0
    classifier.eval()
    for inputs, labels in val_data_loader:
        
        inputs = inputs.type(torch.LongTensor)

        logits = classifier.forward(inputs)
        logits = logits.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)
        
        loss = loss_fn(logits.squeeze(), labels)
        total_loss += loss.item()
    print ('val', total_loss)

accuracy = 0
count = 0

all_pred = []

for inputs, labels in test_data_loader:
    count+=1
        
    inputs = inputs.type(torch.LongTensor)
    logits = classifier.forward(inputs)
        
    predictions = logits.round()
    all_pred.append(predictions.tolist())
    
    #print (torch.sum(predictions == labels))
    
    accuracy += float(torch.sum(predictions == labels))/593

# We use a different method to measure accuracy
savecsv("pred.csv", all_pred)

accuracy = accuracy/count

print (accuracy)
        
