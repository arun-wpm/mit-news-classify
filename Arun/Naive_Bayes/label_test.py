import csv

def loadcsv(filename):
    with open(filename, newline='') as f: 
        return list(csv.reader(f))

true = loadcsv("true.csv")
pred = loadcsv("pred.csv")

freqtrue = {}
freqpred = {}
for i in range(len(true)):
    true[i] = list(true[i])
    pred[i] = list(pred[i])
    for j in range(len(true[i])):
        if j not in freqtrue:
            freqtrue[j] = int(true[i][j])
            freqpred[j] = int(pred[i][j])
        else:
            freqtrue[j] += int(true[i][j])
            freqpred[j] += int(pred[i][j])

for key in freqtrue:
    print(freqtrue[key], freqpred[key])