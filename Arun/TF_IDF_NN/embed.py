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

def embed_p(corpusdir="../../", corpusfile="NYTcorpus.p", vocabfile="small_vocab.csv", logfile="log.txt", outfile="embedded.p"):
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
            for row in all_data:
                data.append(row[2]) # article text

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

        # save the result into another pickle file
        with open(outfile, "wb") as out:
            pickle.dump(data_tfidf, out)
            f.write("Dumped output at " + outfile + "\n")
    except Exception:
        traceback.print_exc(file=f)

    f.close()

#TODO: embed for tsv as well

if __name__ == "__main__":
    embed_p()