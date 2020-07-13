# USAGE EXAMPLE: python compareTags.py NYTtags.csv jiannatags.csv
# By Max Tegmark, May 2020
import csv, sys
#from useful import *

def loadcsv(filename):
    with open(filename, newline='') as f: 
        return list(csv.reader(f))

def savecsv(filename,list):
    with open(filename,"w",newline="") as f:
        csv.writer(f).writerows(list)
        f.flush()

def score(tags1,tags2):
   correct = [t for t in tags2 if t in tags1]
   falseNegatives = [t for t in tags1 if not t in tags2]
   falsePositives = [t for t in tags2 if not t in tags1]
   return (correct,falseNegatives,falsePositives)

def capitalizE(s): return s[:-1]+s[-1].upper()

def compare(tags1,tags2):
    (correct,falseNegatives,falsePositives) = score(tags1,tags2)
    t1 = [s.upper() for s in correct]
    t2 = [s.capitalize() for s in falseNegatives]
    t3 = [capitalizE(s) for s in falsePositives]
    return ([len(t1),len(t2),len(t3)]+t1+t2+t3)

args = sys.argv
if len(args)<3:
    print("Need two arguments.")
    print("USAGE EXAMPLE: python compareTags.py NYTtags.csv jiannatags.csv")
    quit()
(infile1,infile2) = args[1:3]
outfile = "tag_accuracy.csv"

tags1 = loadcsv(infile1)
print("Tags loaded for",len(tags1),"articles from",infile1)
tags2 = loadcsv(infile2)
print("Tags loaded for",len(tags2),"articles from",infile2)
n = min(len(tags1),len(tags2))
comparison = [compare(tags1[i],tags2[i]) for i in range(n)]
savecsv(outfile,comparison)
print("Comparisons for",n,"arcicles saved in",outfile)


correct = sum([s[0] for s in comparison])
falseNegatives = sum([s[1] for s in comparison])
falsePositives = sum([s[2] for s in comparison])
ntags = correct + falseNegatives
print("Correct tags/articls........",correct/n)
print("False negatives/article.....",falseNegatives/n)
print("False positves/article......",falsePositives/n)
print("===========================================")
print("NYT tags/article............",ntags/n)
print("False negatives/tag.........",falseNegatives/ntags)
print("False positves/tag..........",falsePositives/ntags)
