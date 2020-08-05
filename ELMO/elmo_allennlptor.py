# job [1] 67846 7/22
# job [1] 67983 7/22
# [1] 69917 7/22
# [1] 70104 7/22
# [1] 70197 7/22
# [1] 70220 7/22
import pandas as pd
import numpy as np

from spacy.lang.en import English

import torch
from allennlp.modules.elmo import Elmo, batch_to_ids


nlp = English()
# Create a Tokenizer with the default settings for English
# including punctuation rules and exceptions
tokenizer = nlp.Defaults.create_tokenizer(nlp)

def tokenize_str(text: str, tokenizer):
    tokenized_str = map(str, list(tokenizer(text)))
    return tokenized_str

options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
elmo = Elmo(options_file, weight_file, 1, dropout=0)

def elmo_embed_words(tokenized, elmo):
    character_ids = batch_to_ids(tokenized)
    embeddings = elmo(character_ids)
    
    return embeddings['elmo_representations'][0]

def average_embedding_from_embedding_list(embedding):
    numnonzero = (embedding.sum(axis=-1) != 0).sum()
    summedembedding = embedding.sum(axis=0).sum(axis=0)
    return summedembedding / numnonzero

def average_embedding_of_text(text: str, tokenizer, elmo):
    tokenized_str = tokenize_str(text, tokenizer)
    elmo_wordembeddings = elmo_embed_words(tokenized_str, elmo)
    return average_embedding_from_embedding_list(elmo_wordembeddings).detach().numpy()


# nyt_list = pd.read_pickle('../../NYTcorpus_updated.p')
# nyt_np = np.array(nyt_list)  # this actually ends up being 1d, the individual items remain lists
# nyt_df = pd.DataFrame(nyt_list)
nyt_df = pd.DataFrame(pd.read_pickle('../../NYTcorpus_updated.p'))

# start = 1574
start = 1778
it = 0
for text in nyt_df[2][start:]:
    textembed_df = pd.DataFrame(average_embedding_of_text(text, tokenizer, elmo)).T
    textembed_df.to_csv('nytcorpuselmoallen.csv', mode='a', header=None, index=False)
    print(start + it)
    