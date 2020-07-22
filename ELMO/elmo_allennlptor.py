#!/usr/bin/env python
# coding: utf-8

# In[94]:


import pandas as pd
import numpy as np

from spacy.lang.en import English

import torch
from allennlp.modules.elmo import Elmo, batch_to_ids


# In[220]:


# from spacy.lang.en import English
nlp = English()
# Create a Tokenizer with the default settings for English
# including punctuation rules and exceptions
tokenizer = nlp.Defaults.create_tokenizer(nlp)

def tokenize_str(text: str, tokenizer):
    tokenized_str = map(str, list(tokenizer(text)))
    return tokenized_str

# options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
# weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
# elmo = Elmo(options_file, weight_file, 1, dropout=0)

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


# In[21]:


# nyt_list = pd.read_pickle('../../NYTcorpus_updated.p')
# nyt_np = np.array(nyt_list)  # this actually ends up being 1d, the individual items remain lists
# nyt_df = pd.DataFrame(nyt_list)
nyt_df = pd.DataFrame(pd.read_pickle('../../NYTcorpus_updated.p'))


# In[235]:


for text in nyt_df[2][300:1000]:
    textembed_df = pd.DataFrame(average_embedding_of_text(text, tokenizer, elmo)).T
    textembed_df.to_csv('nytcorpuselmoallen.csv', mode='a', header=None, index=False)


# In[ ]:





# In[ ]:





# In[164]:


happy = smallnytdf[2].apply(lambda text: elmo_embed_words(list(tokenize_str(text, tokenizer)), elmo))


# In[226]:


pd.DataFrame(np.array([1,2,3,4])).T


# In[193]:


finalsmallnytdf = happy.apply(lambda embeddings: average_embedding_from_embedding_list(embeddings))


# In[225]:


np.expand_dims(finalsmallnytdf[0].detach().numpy(), axis=0)


# In[206]:


np.array(list(finalsmallnytdf.apply(lambda pytensor: pytensor.detach().numpy().tolist())))


# In[201]:


finalsmallnytdf.apply(lambda pytensor: list(pytensor))


# In[177]:


(finalsmallnytdf[0])


# In[178]:


finalsmallnytdf[1]


# In[179]:


finalsmallnytdf[2]


# In[180]:


finalsmallnytdf[3]


# In[165]:


list(map(lambda x: x.shape, happy))


# In[174]:


(happy[0].sum(axis=-1) != 0).sum()


# In[167]:


happy[1]


# In[168]:


happy[0] == happy[1]


# In[122]:


options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

# Compute two different representation for each token.
# Each representation is a linear weighted combination for the
# 3 layers in ELMo (i.e., charcnn, the outputs of the two BiLSTM))
elmo = Elmo(options_file, weight_file, 1, dropout=0)

# use batch_to_ids to convert sentences to character ids
sentences = [['First', 'sentence', '.'], ['Another', '.']]
character_ids = batch_to_ids(sentences)

embeddings = elmo(character_ids)
embeddings

# embeddings['elmo_representations'] is length two list of tensors.
# Each element contains one layer of ELMo representations with shape
# (2, 3, 1024).
#   2    - the batch size
#   3    - the sequence length of the batch
#   1024 - the length of each ELMo vector


# In[123]:


# tenstest = 
embeddings['elmo_representations'][0]


# In[121]:


tenstest.sum(axis=0).sum(axis=0) 


# In[103]:


tenstest


# In[110]:





# In[ ]:





# In[14]:


smallnytdf = nyt_df.head()


# In[71]:


tokenizer = SpacyTokenizer("en_core_web_sm")


# In[86]:


# from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
# tokenizer = SpacyTokenizer("en_core_web_sm")
# tokenizer.tokenize(nyt_df.iloc[1,2])

# from spacy.lang.en import English
# nlp = English()
# # Create a Tokenizer with the default settings for English
# # including punctuation rules and exceptions
# tokenizer = nlp.Defaults.create_tokenizer(nlp)

# nyt_tokenized = list(tokenizer(nyt_df.iloc[1,2]))

# from nltk.tokenize import WordPunctTokenizer
# WordPunctTokenizer().tokenize(nyt_df.iloc[1,2])


# In[45]:


nyt_charids = batch_to_ids(nyt_df.iloc[1,2])


# In[76]:


nyt_df.iloc[1,2]


# In[ ]:





# In[79]:


nyt_charids = batch_to_ids(map(str, nyt_tokenized))


# In[80]:





# In[ ]:




