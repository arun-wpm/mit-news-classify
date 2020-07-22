"""
Created Thursday June 18 2020 20:53 +0700

@author: arunwpm
"""
import traceback
import csv
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import tokenize

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

def embed_p(corpusdir="../../", 
            corpusfile="NYTcorpus.p", 
            logfile="log.txt", 
            outfile="embedded_100%.p", 
            labelfile="binarylabels_100%.p", 
            labeldictfile="labelsdict_100%.p"):
    # embed each article in a pickle format corpus into a singular vector using vocabfile as the vocabulary
    # the format of the corpus is assumed to be the same as NYTcorpus.p

    f = open(logfile, 'a+')
    f.write("DOC2VEC_EMBED_P\n")
    try:
        # load the whole corpus
        with open(corpusdir + corpusfile, "rb") as corpus:
            all_data = pickle.load(corpus)
            f.write("Loaded corpus!\n")
            data = []
            labels = []
            l = len(all_data)
            for i in range(0, l): # memory errors rip
                article = all_data[i][2]
                # tokenize each article
                data.append(TaggedDocument(list(tokenize(article)), [i])) # article text
                labels.append(all_data[i][3:]) # article labels
            label_matrix, label_dict = transform_labels(labels) # make labels binary
        f.write("Words are tokenized\n")

        # transform data by doc2vec
        model = Doc2Vec(data, vector_size=100, window=5, workers=12, epochs=10)
        model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
        vec_data = []
        for i in range(0, l):
            vec_data.append(model.docvecs[i])

        # transform the test data by tf_idf as well
        f.write("Doc2Vec done\n")

        # save the results into pickle files
        with open(outfile, "wb") as out:
            pickle.dump(vec_data, out)
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
    embed_p()
