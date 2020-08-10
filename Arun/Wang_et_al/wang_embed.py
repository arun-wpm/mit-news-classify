"""
Created Tuesday June 16 2020 15:43 +0700

@author: arunwpm
"""
# research paper: https://arxiv.org/pdf/1805.04174.pdf
# adapted to use BERT instead of GloVe for the embedding

import torch
from transformers import BertTokenizer, BertModel
import traceback
import csv
import pickle

def loadcsv(filename):
    with open(filename, newline='') as f: 
        return list(csv.reader(f))

def savecsv(filename, list):
    with open(filename, "w", newline="") as f:
        csv.writer(f).writerows(list)

def transform_labels(in_labels):
    # from the input labels, in the format of list of list of labels pertaining to document at row i,
    # output a multilabel matrix where row i column j is 1 if document i contains label j and 0 otherwise
    label_to_id = {}
    id_to_label = {}
    cnt = 0
    label_matrix = []
    for row in in_labels:
        label_matrix.append([])
        for label in row:
            if label not in label_to_id:
                label_to_id[label] = cnt
                id_to_label[cnt] = label
                cnt += 1
            i = label_to_id[label]
            if i >= len(label_matrix[-1]):
                label_matrix[-1].extend([0]*(i + 1 - len(label_matrix[-1])))
            label_matrix[-1][i] = 1
    
    for i in range(len(label_matrix)):
        if len(label_matrix[i]) < len(label_to_id):
            label_matrix[i].extend([0]*(len(label_to_id) - len(label_matrix[i])))

    return label_matrix, id_to_label

def wang_embed_p(corpusdir="../../", 
                 corpusfile="NYTcorpus.p", 
                 id2tagloc="nyt-theme-tags.csv",
                 logfile="log.txt", 
                 outfile="embedded.p", 
                 labelfile="binarylabels.p", 
                 labeldictfile="labelsdict.p",
                 tfmerfile="tfmer.p"):
    # embed each article in a pickle format corpus into a singular vector using Wang et al's model
    # the format of the corpus is assumed to be the same as NYTcorpus.p

    f = open(logfile, 'a+')
    f.write("WANG_EMBED_P\n")
    try:
        # load the whole corpus
        with open(corpusdir + corpusfile, "rb") as corpus:
            all_data = pickle.load(corpus)
            f.write("Loaded corpus!\n")
            data = []
            labels = []
            for row in all_data:
                data.append(row[2]) # article text
                labels.append(row[3:]) # article labels
            label_matrix, label_dict = transform_labels(labels) # make labels binary

        # initialize the id2tag dictionary (key: tag id, value: tag name - important in this case!)
        id2tag_table = loadcsv(id2tagloc)
        id2tag = {}
        for row in id2tag_table:
            if row == []:
                continue
            id2tag[row[1]] = row[2]

        # f0: X |-> V, the text sequence is represented as its word-embedding form V, which is a matrix of P * L.
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        model = BertModel.from_pretrained('bert-base-cased')

        C = torch.FloatTensor()
        for i in range(len(label_dict)):
            tag = id2tag[label_dict[i]]
            input_ids = torch.tensor(tokenizer.encode(tag, add_special_tokens=True)).unsqueeze(0)
            out = model(input_ids)
            # print(out) #debugging I guess
            C = torch.cat((C, out[0]), 0)
        # shape of C: K=5xx, n, 768

        cos = torch.nn.CosineSimilarity(dim=2)
        for row in data:
            input_ids = torch.tensor(tokenizer.encode(row, add_special_tokens=True)).unsqueeze(0)
            out = model(input_ids)
            # print(out) #debugging I guess
            # shape of out[0]: 1, L, 768
            sim = cos(out[0], )

            


        # f1: V |-> z, a compositional function f1 aggregates word embeddings into a fixed-length vector representation z.

        # f2: z |-> y, a classifier f2 annotates the text representation z with a label
        # I will use ../TF_IDF_NN/nn.py for this stage

        # tokenize each article and count words in data given a given vocabulary
        f.write("Words are tokenized by the almighty BERT\n")

        # transform the test data by tf_idf as well
        f.write("TF-IDF done:" + str(data_tfidf.shape) + '\n')

        # save the results into pickle files
        with open(outfile, "wb") as out:
            pickle.dump(data_tfidf, out)
            f.write("Dumped data output at " + outfile + "\n")
        
        with open(labelfile, "wb") as label:
            pickle.dump(label_matrix, label)
            f.write("Dumped label output at " + labelfile + "\n")

        with open(labeldictfile, "wb") as labeldict:
            pickle.dump(label_dict, labeldict)
            f.write("Dumped label dictionary at " + labeldictfile + "\n")
    
    except Exception:
        traceback.print_exc(file=f)

    f.close()

if __name__ == "__main__":
    wang_embed_p()