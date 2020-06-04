"""
Created Thursday June 4 2020 18:41 +0700

@author: arunwpm
"""
import csv

def loadcsv(filename):
    with open(filename, newline='') as f: 
        return list(csv.reader(f))

def savecsv(filename, list):
    with open(filename, "w", newline="") as f:
        csv.writer(f).writerows(list)

def buildVocab(keywordloc="keywords.csv"):
    keywords = loadcsv(keywordloc)
    vocab = set()
    for row in keywords:
        print(row)
        for kw in row[1:]:
            print(kw)
            print(kw.strip('()').split(' ')[0].strip("',"))
            vocab.add(kw.strip('()').split(' ')[0].strip("',"))
    savecsv("small_vocab.csv", [[x] for x in vocab])