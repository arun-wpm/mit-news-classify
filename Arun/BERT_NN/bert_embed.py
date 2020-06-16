"""
Created Tuesday June 16 2020 19:35 +0700

@author: arunwpm
"""
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

def bert_embed_p(corpusdir="../../", 
                 corpusfile="NYTcorpus.p", 
                 logfile="log.txt", 
                 outfile="embedded.p", 
                 labelfile="binarylabels.p", 
                 labeldictfile="labelsdict.p"):
    # embed each article in a pickle format corpus into a singular vector using Wang et al's model
    # the format of the corpus is assumed to be the same as NYTcorpus.p

    f = open(logfile, 'a+')
    f.write("BERT_EMBED_P\n")
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

        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        model = BertModel.from_pretrained('bert-base-cased')

        X = torch.FloatTensor()
        for row in data:
            input_ids = torch.tensor(tokenizer.encode(row, add_special_tokens=True)).unsqueeze(0)
            out = model(input_ids)
            # print(out) #debugging I guess
            X = torch.cat((X, torch.mean(out[0], dim=1)), 0)
        # shape of X: N, <= 512, 768
        f.write("Words are tokenized and embedded by the almighty BERT\n")

        # save the results into pickle files
        with open(outfile, "wb") as out:
            pickle.dump(X.detach().numpy(), out)
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
    bert_embed_p()