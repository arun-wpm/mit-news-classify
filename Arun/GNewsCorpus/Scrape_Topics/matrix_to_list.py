import csv
from random import shuffle

def loadtsv(filename):
    with open(filename, newline='') as f: 
        return list(csv.reader(f,delimiter="\t"))

def savecsv(filename,list):
    with open(filename,"w",newline="") as f:
        csv.writer(f).writerows(list)
        f.flush()

if __name__ == "__main__":
    data = loadtsv("media_matrix.tsv")
    flat = []
    r = len(data)
    c = len(data[0])
    for i in range(1, c):
        cnt = 0
        for j in range(1, r):
            if (len(data[j][i]) > 0):
                cnt += 1
                flat.append([data[j][0]+"_"+data[0][i], data[j][i]])
        print(cnt)
    shuffle(flat)
    savecsv("medialist2shuf.csv", flat)