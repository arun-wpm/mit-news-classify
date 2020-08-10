import os
import csv

def loadcsv(filename):
    with open(filename, newline='') as f: 
        return list(csv.reader(f))

def loadtsv(filename):
    with open(filename, newline='') as f: 
        return list(csv.reader(f,delimiter="\t"))

def savetsv(filename,list):
    with open(filename, "w", newline="") as f:
        csv.writer(f,delimiter="\t").writerows(list)
        f.flush()

if __name__ == "__main__":
    dir = ""
    urls = []
    for filename in os.listdir(dir):
        if "articles" in filename:
            data = loadtsv(filename)
            urls.extend([[row[0]] for row in data])
        if "schedule" in filename:
            data = loadtsv(filename)
            for row in data:
                urls.extend([[x[0]] for x in row[4:] if type(x) == tuple else [x]])
    savetsv("urls_crawled_to_date.tsv", urls)