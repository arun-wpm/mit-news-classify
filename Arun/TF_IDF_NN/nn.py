import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle

def init(corpusfile="../../NYTcorpus.p", embeddedfile="embedded.p"):
    # load the input
    with open(inputfile, "rb") as f:
        data = pickle.load(f)
    
    # set up the model
    model = keras.Sequential()
    #TODO