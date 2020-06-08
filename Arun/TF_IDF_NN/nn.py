import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import random
import numpy as np
import csv

def loadcsv(filename):
    with open(filename, newline='') as f: 
        return list(csv.reader(f))

def savecsv(filename, list):
    with open(filename, "w", newline="") as f:
        csv.writer(f).writerows(list)

def init(embeddedfile="embedded.p", binarylabelsfile="binarylabels.p"):
    print("Initializing model based on " + embeddedfile + ", " + binarylabelsfile)
    # load the input
    with open(embeddedfile, "rb") as f:
        data = pickle.load(f)
    with open(binarylabelsfile, "rb") as g:
        label = pickle.load(g)
    
    # set up the model
    model = keras.Sequential()
    model.add(keras.Input(shape=(data.shape[1],)))
    model.add(layers.Dense(400, activation="relu"))
    model.add(layers.Dense(200, activation="relu"))
    model.add(layers.Dense(len(label[0]), activation="sigmoid")) # can't do softmax because it's multilabel
    model.summary()

    return data, label, model

train_ratio = 0.75
val_ratio = 0.10
test_ratio = 0.15

def split_data(data, label, model):
    print("Splitting data...")
    s1 = int(train_ratio*data.shape[0])
    s2 = int((train_ratio + val_ratio)*data.shape[0])
    #TODO: shuffle the data and labels
    # data should now be an nparray of float64
    data_train = data[:s1].toarray()
    data_val = data[s1:s2].toarray()
    data_test = data[s2:].toarray()
    label_train = np.array(label[:s1], dtype=np.float64)
    label_val = np.array(label[s1:s2], dtype=np.float64)
    label_test = np.array(label[s2:], dtype=np.float64)
    return data_train, data_val, data_test, label_train, label_val, label_test

def train(model, data_train, data_val, label_train, label_val):
    print("Training model...")
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryCrossentropy()],
    )
    hist = model.fit(
        data_train,
        label_train,
        batch_size=64,
        epochs=2, # technically I should run this until validation accuracy peaks but I don't know how long things take yet
        validation_data=(data_val, label_val),
    )

def predict(model, data_test):
    print("Predicting data...")
    pred = model.predict(data_test)
    return pred

def score(true, pred):
    #using the same scoring method as in jiannatags.py
    correct = 0
    falseneg = 0
    falsepos = 0
    for i in range(len(true)):
        for j in range(len(true[i])):
            if true[i][j] == pred[i][j]:
                correct += 1
            else:
                if true[i][j] > pred[i][j]:
                    falseneg += 1
                else:
                    falsepos += 1
    
    print("Correct Rate " + str(correct/len(true)) + '\n')
    print("False Negative Rate " + str(falseneg/len(true)) + '\n')
    print("False Positive Rate " + str(falsepos/len(true)) + '\n')

if __name__ == "__main__":
    data, label, model = init()
    data_train, data_val, data_test, label_train, label_val, label_test = split_data(data, label, model)
    train(model, data_train, data_val, label_train, label_val)
    pred = predict(model, data_test)
    score(label_test, pred)

    # save some outputs
    savecsv("true.csv", label_test)
    savecsv("pred.csv", pred)