import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import random
import numpy as np
import csv
from scipy.sparse import csr_matrix

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
        label = csr_matrix(pickle.load(g), dtype=np.float64)
    
    # set up the model
    model = keras.Sequential()
    model.add(keras.Input(shape=(data.shape[1],)))
    model.add(layers.Dense(400, activation="relu"))
    model.add(layers.Dense(200, activation="relu"))
    model.add(layers.Dense(label.shape[1], activation="sigmoid")) # can't do softmax because it's multilabel
    model.summary()

    return data, label, model

train_ratio = 0.75
val_ratio = 0.10
test_ratio = 0.15

batch_size = 64

class DataGenerator(keras.utils.Sequence):
    def __init__(self, data, label, size, batch_size=64):
        self.batch_size = batch_size
        self.data = data
        self.size = size
        self.label = label
    
    def on_epoch_end(self):
        shuf = np.arange(self.size)
        np.random.shuffle(shuf)
        self.data = self.data[shuf, :]
        self.label = self.label[shuf, :]
    
    def __len__(self):
        return int(np.floor((self.size)/self.batch_size))
    
    def __getitem__(self, index):
        if self.size < (index+1)*self.batch_size:
            return (self.data[index*self.batch_size : ].toarray(),
                    self.label[index*self.batch_size : ].toarray())
        else:
            return (self.data[index*self.batch_size : (index+1)*batch_size].toarray(),
                    self.label[index*self.batch_size : (index+1)*batch_size].toarray())

def split_data(data, label):
    print("Splitting data...")
    s1 = int(train_ratio*data.shape[0])
    s2 = int((train_ratio + val_ratio)*data.shape[0])
    # initial shuffle the data and labels
    shuf = np.arange(data.shape[0])
    np.random.shuffle(shuf)
    data = data[shuf, :]
    label = label[shuf, :]
    gen_train = DataGenerator(data[:s1], label[:s1], s1, batch_size)
    data_val = data[s1:s2].toarray() # no generator support for validation :(
    label_val = label[s1:s2].toarray()
    data_test = data[s2:].toarray()
    label_test = label[s2:].toarray()
    return gen_train, data_val, label_val, data_test, label_test

    # data should now be an nparray of float64
    # it is important to note that this one only uses 1% of the NYT corpus
    # data_train = data[:s1:100].toarray()
    # data_val = data[s1:s2:100].toarray()
    # data_test = data[s2::100].toarray()
    # label_train = np.array(label[:s1:100], dtype=np.float64)
    # label_val = np.array(label[s1:s2:100], dtype=np.float64)
    # label_test = np.array(label[s2::100], dtype=np.float64)
    # return data_train, data_val, data_test, label_train, label_val, label_test

def train(model, gen_train, data_val, label_val):
    print("Training model...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.00001),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryCrossentropy()],
    )
    hist = model.fit(
        gen_train,
        # data_train,
        # label_train,
        batch_size=batch_size,
        epochs=30, # technically I should run this until validation accuracy peaks but I don't know how long things take yet
        validation_data=(data_val, label_val),
        use_multiprocessing=True,
    )

def predict(model, gen_test):
    print("Predicting data...")
    pred = model.predict(
        gen_test,
        batch_size=batch_size,
        use_multiprocessing=True,
    )
    return pred

def score(true, pred):
    # using the same scoring method as in jiannatags.py
    correct = 0
    falseneg = 0
    falsepos = 0
    for i in range(len(true)):
        for j in range(len(true[i])):
            # if true[i][j] == pred[i][j]:
            if true[i][j] == (pred[i][j] >= 0.5): # sigmoid woohoo
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
    gen_train, data_val, label_val, data_test, label_test = split_data(data, label)
    train(model, gen_train, data_val, label_val)
    pred = predict(model, data_test)
    score(label_test, pred)

    # save some outputs
    savecsv("true.csv", label_test)
    savecsv("pred.csv", pred)
    model.save("model.h5")