import sys
import csv

def loadcsv(filename):
    with open(filename, newline='') as f: 
        return list(csv.reader(f))

def score(true, pred):
    trueneg = 0
    truepos = 0
    falseneg = 0
    falsepos = 0
    for i in range(len(true)):
        for j in range(len(true[i])):
            true[i][j] = float(true[i][j])
            pred[i][j] = float(pred[i][j])
            if true[i][j] == (pred[i][j] >= 0.5): # threshold can be changed
                if true[i][j] > 0:
                    truepos += 1
                else:
                    trueneg += 1
            else:
                if true[i][j] > pred[i][j]:
                    falseneg += 1
                else:
                    falsepos += 1
    
    print("Correct " + str(truepos) + "(True Positive) " + str(trueneg) + "(True Negative)\n")
    print("False Negative " + str(falseneg) + '\n')
    print("False Positive " + str(falsepos) + '\n')

    print("Precision " + str(truepos/(truepos+falsepos)) + '\n')
    print("Recall " + str(truepos/(truepos+falseneg)) + '\n')

if __name__ == "__main__":
    # input the binary matrices
    true = loadcsv(sys.argv[1])
    pred = loadcsv(sys.argv[2])
    score(true, pred)