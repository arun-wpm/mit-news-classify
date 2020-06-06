#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 00:31:33 2020

@author: haimoshri
"""

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
epochs = 10
vocab_size = 0
loss_fn = nn.MultiLabelSoftMarginLoss()

def load_data(filename,  vocab_size):
    with open(filename, 'r') as f:
        file = pd.read_csv(f)
        head_tag = []
        for i in range(593):
            head_tag.append(str(i))
        
        num_tags = rt.num_tags
        
        vocabulary = dp.build_vocabulary(file)
        
        vocab_size = len(vocabulary)
        
        td_matrix = dp.build_bow(vocabulary, file)
        
        labels = dp.build_labels(num_tags, file)
        
        return td_matrix, labels
        
def prepare_data(td_matrix, labels):
    
        split_1 = int(4*len(td_matrix)/5)
        split_2 = int(9*len(td_matrix)/10)
        
        
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

td_matrix, labels = load_data('../Data/small_data_test.csv',  vocab_size)

train_data_loader, val_data_loader, test_data_loader = prepare_data(td_matrix, labels)

class MyModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, output_size, num_layers, dropout=0):
        super(MyModel, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout, bidirectional=False)
        self.output = nn.Linear(hidden_size, output_size)
        #self.activation = nn.Sigmoid()


    def forward(self, x):
        embeds = self.embedding(x)
        outputs1, a = self.lstm(embeds)
        outputs = self.output(outputs1)
        outputs = outputs[:, -1, :]
        #outputs = self.activation(outputs)
        return outputs

classifier = MyModel(vocab_size, embedding_size, hidden_size, output_size, num_layers, dropout)
optimizer = optim.SGD(classifier.parameters(), lr=lr, momentum=0.9)

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
    #print ('train', total_loss)
    
    total_loss = 0
    classifier.eval()
    for inputs, labels in val_data_loader:
        
        inputs = inputs.type(torch.LongTensor)

        logits = classifier.forward(inputs)
        logits = logits.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)
        
        loss = loss_fn(logits.squeeze(), labels)
        total_loss += loss.item()
    #print ('val', total_loss)

accuracy = 0
count = 0

for inputs, labels in test_data_loader:
    count+=1
        
    inputs = inputs.type(torch.LongTensor)
    logits = classifier.forward(inputs)
        
    predictions = logits.round()
    
    print (torch.sum(predictions == labels))
    
    accuracy += float(torch.sum(predictions == labels))/593

accuracy = accuracy/count

print (accuracy)
        