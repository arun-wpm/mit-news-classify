"""
Created Wednesday August 12 2020 19:40 +0700

@author: arunwpm
"""
# https://medium.com/@adriensieg/text-similarities-da019229c894
# Word2Vec + Smooth Inverse Frequency + Cosine Similarity
import traceback
import csv
import pickle
import numpy as np
from scipy import spatial
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
            outfile="embedded_cos.p", 
            labelfile="binarylabels_cos.p", 
            labeldictfile="labelsdict_cos.p",
            id2tagloc = 'nyt-theme-tags.csv' #name of the conversion table from tag id to tag name for NYTcorpus
            ):
    # embed each article in a pickle format corpus into a singular vector using vocabfile as the vocabulary
    # the format of the corpus is assumed to be the same as NYTcorpus.p

    # f = open(logfile, 'a+')
    print("DOC2VEC_EMBED_P\n")
    try:
        # load the whole corpus
        with open(corpusdir + corpusfile, "rb") as corpus:
            all_data = pickle.load(corpus)
            print("Loaded corpus!\n")
            data = []
            textdata = []
            labels = []
            l = len(all_data)
            for i in range(0, l): # memory errors rip
                article = all_data[i][2]
                # tokenize each article
                data.append(TaggedDocument(list(tokenize(article)), [i])) # article text
                textdata.append(article) # article text
                labels.append(all_data[i][3:]) # article labels
            label_matrix, label_dict = transform_labels(labels) # make labels binary
        print("Words are tokenized\n")

        # tokenize each article and count words in data
        count_vect = CountVectorizer()
        data_counts = count_vect.fit_transform(textdata)
        print("Vocabulary built\n")

        # transform data by tf_idf
        # freq = np.sum(data_counts.toarray(), axis=0)
        freq = data_counts.sum(axis=0)
        total = np.sum(freq)

        # sif
        sif = {}
        a = 0.001
        for vocab, index in count_vect.vocabulary_.items():
            sif[vocab] = a/(a + freq[0,index]/total)

        # transform data by doc2vec
        model = Doc2Vec(data, vector_size=300, window=5, workers=12, epochs=10)
        model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

        # vector representation of labels
        # ld = loadcsv(ldloc)
        id2tag_table = loadcsv(id2tagloc)
        id2tag = {}
        for row in id2tag_table:
            if row == []:
                continue
            id2tag[row[1]] = list(tokenize(row[2], lowercase=True))
        label_vec_data = []
        for i in range(len(label_dict)):
            vec = np.zeros(100)
            for word in id2tag[label_dict[i]]:
                vec += model.wv.get_vector(word)*sif[word]
            vec /= len(id2tag[label_dict[i]])
            label_vec_data.append(vec)

        vec_data = []
        for i in range(0, l):
            vec = np.zeros(100)
            for word in data[i].words:
                vec += model.wv.get_vector(word)*sif[word]
            vec /= len(data[i].words)
            #cosine similarity for each of the labels (this is quite slow?)
            cos = np.zeros(len(label_vec_data))
            for j in range(len(label_vec_data)):
                cos[j] = 1 - spatial.distance.cosine(vec, label_vec_data[j])
            vec_data.append(cos)
        print("Doc2Vec + SIF + cosine similarity done\n")

        # save the results into pickle files
        with open(outfile, "wb") as out:
            pickle.dump(vec_data, out)
            print("Dumped data output at " + outfile + "\n")
        
        with open(labelfile, "wb") as label:
            pickle.dump(label_matrix, label)
            print("Dumped label output at " + labelfile + "\n")

        with open(labeldictfile, "wb") as labeldict:
            pickle.dump(label_dict, labeldict)
            print("Dumped label dictionary at " + labeldictfile + "\n")
        
    except Exception:
        traceback.print_exc(file=f)

    # f.close()

if __name__ == "__main__":
    embed_p()