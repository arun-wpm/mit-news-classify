"""
Created Thursday June 11 2020 19:24 +0700

@author: arunwpm
"""
import sys
import csv
import os
import pickle
import re

def loadtsv(filename):
    with open(filename, newline='') as f: 
        return list(csv.reader(f,delimiter="\t"))

#directory of news articles that have been scraped
newsdir = "../../../MORE_NEWS/"

if __name__ == '__main__':
    keyword = sys.argv[1] #first argument: the keyword to search for (maybe the topic? for example abortion)
    sources = sys.argv[2:] #other arguments: sources to use
    print("Keyword is: " + keyword)
    all_data = []
    for filename in os.listdir(newsdir):
        #loops through each file in the directory, and extract articles which have the keyword
        print("Going through " + filename + "...")
        newspaper = filename.split("_")[1]
        if (newspaper not in sources and sources != []):
            continue
        all_articles = loadtsv(newsdir + filename)
        for article in all_articles:
            if (len(article) < 7):
                print("idk what's wrong...")
                continue
            if re.search(keyword, article[6], re.IGNORECASE):
                all_data.append([[], [], article[6], newspaper])
    with open("allnews_" + keyword + "_".join(sources) + ".p", "wb") as out:
        pickle.dump(all_data, out)
    print("Done, data saved at allnews.p")
