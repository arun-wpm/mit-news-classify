#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 11:57:38 2020

@author: haimoshri
"""

from transformers import BertModel, BertTokenizer

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


tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

max_len = 64
batch_size = 64
epochs = 3
lr = 0.01
loss_fn = nn.MultiLabelMarginLoss()

HIDDEN_SIZE = 768 
NUM_LABEL = 593

def process_data(filename, tokenizer, max_len):
    with open(filename, 'r') as f:
        file = pd.read_csv(f)
        
        result = np.zeros((len(file['Text']), max_len))
        attention_mask = np.zeros((len(file['Text']), max_len))
        
        count = 0
        
        for text in file['Text']:
            encoded_dict = tokenizer.encode_plus(text, add_special_tokens = True, max_length = max_len, pad_to_max_length = True,
                                                 return_attention_mask = True, return_tensors = 'pt',)
            result[count, :] = encoded_dict['input_ids'] 
            attention_mask[count, :] = encoded_dict['attention_mask']
    
            count+=1
        
        num_tags = rt.num_tags
        labels = dp.build_labels(num_tags, file)
        
        return result, attention_mask, labels

def prepare_data(filename, tokenizer, max_len, batch_size):
    result, attention_mask, labels = process_data(filename, tokenizer, max_len)
    
    split_1 = int(len(result)*0.75)
    split_2 = int(len(result)*0.85)
    
    train_data_processed = result[:split_1, :]
    train_data_mask = attention_mask[:split_1, :]
    train_labels = labels[:split_1, :]
    
    val_data_processed = result[split_1:split_2, :]
    val_data_mask = attention_mask[split_1:split_2, :]
    val_labels = labels[split_1:split_2, :]
    
    test_data_processed = result[split_2:, :]
    test_data_mask = attention_mask[split_2:, :]
    test_labels = labels[split_2:, :]
    
    train_data_tensor = TensorDataset(torch.from_numpy(train_data_processed), torch.from_numpy(train_data_mask), torch.from_numpy(train_labels))
    val_data_tensor = TensorDataset(torch.from_numpy(val_data_processed), torch.from_numpy(val_data_mask), torch.from_numpy(val_labels))
    test_data_tensor = TensorDataset(torch.from_numpy(test_data_processed), torch.from_numpy(test_data_mask), torch.from_numpy(test_labels))
    
    train_data_loader = DataLoader(train_data_tensor, shuffle=True, batch_size=batch_size)
    val_data_loader = DataLoader(val_data_tensor, shuffle=True, batch_size=batch_size)
    test_data_loader = DataLoader(test_data_tensor, shuffle=True, batch_size=1)
    
    return train_data_loader, val_data_loader, test_data_loader
    

train_data_loader, val_data_loader, test_data_loader = prepare_data('../Data/small_data_test.csv', tokenizer, max_len, batch_size) 
    
class MyBertModel(nn.Module):
    def __init__(self, in_size, out_size):
        super(MyBertModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.linear = nn.Linear(in_size, out_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, inputs, masks):
        output = self.bert(inputs, masks)
        self.hidden_size = output[0].shape[-1]
        output = torch.mean(output[0], 1)
        output = self.linear(output)
        output = self.sigmoid(output)
        return output


model = MyBertModel(HIDDEN_SIZE, NUM_LABEL)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

for epoch in range(epochs):
    
    total_loss = 0
    model.train()
    for inputs, masks, labels in train_data_loader:
    
        inputs = inputs.type(torch.LongTensor)

        model.zero_grad()
        
        output = model.forward(inputs, masks)
        
        labels = labels.type(torch.LongTensor)
        
        loss = loss_fn(output, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    print ('train', total_loss)
    
    total_loss = 0
    model.eval()
    for inputs, masks, labels in val_data_loader:
        
        inputs = inputs.type(torch.LongTensor)

        output = model.forward(inputs, masks)
        
        labels = labels.type(torch.LongTensor)
        
        loss = loss_fn(output, labels)
        total_loss += loss.item()
    print ('val', total_loss)
    
accuracy = 0
count = 0
    
for inputs, masks, labels in test_data_loader:
    count+=1
        
    inputs = inputs.type(torch.LongTensor)
    logits = model.forward(inputs, masks)
        
    predictions = logits.round()
    
    #print (torch.sum(predictions == labels))
    
    accuracy += float(torch.sum(predictions == labels))/593

accuracy = accuracy/count

print (accuracy)
    