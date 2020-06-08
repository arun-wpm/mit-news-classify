"""
Created Friday June 5 2020 17:19 +0700

@author: arunwpm
"""
import traceback
import csv
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

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

def embed_p(corpusdir="../../", corpusfile="NYTcorpus.p", vocabfile="small_vocab.csv", logfile="log.txt", outfile="embedded.p", labelfile="binarylabels.p", labeldictfile="labelsdict.p"):
    # embed each article in a pickle format corpus into a singular vector using vocabfile as the vocabulary
    # the format of the corpus is assumed to be the same as NYTcorpus.p

    f = open(logfile, 'a+')
    f.write("EMBED_P\n")
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

        # tokenize each article and count words in data given a given vocabulary
        vocab = loadcsv(vocabfile)
        vocab = [x[0] for x in vocab] #extra layer of list that we don't want here
        count_vect = CountVectorizer(vocabulary=vocab)
        data_counts = count_vect.fit_transform(data)
        f.write("Words are tokenized, vocabulary built from " + vocabfile + "\n")

        #TODO: use other methods as well, BOW for instance

        # transform data by tf_idf
        tfmer = TfidfTransformer(sublinear_tf=True) #by default
        data_tfidf = tfmer.fit_transform(data_counts)

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

#TODO: embed for tsv as well

if __name__ == "__main__":
    embed_p()