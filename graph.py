import re
import matplotlib.pyplot as plt
import math
import csv

def loadcsv(filename):
    with open(filename, newline='') as f: 
        return list(csv.reader(f))

def findcolor(model):
    color = {
        'BERT + NN':'#500000',
        'BERT + LSTM RNN':'#500000',
        'DistillBERT + NN':'#000050',
        'doc2vec + NN':'#000000',
        'RUM':'#FF0000',
        'TF-IDF + NB':'#105090',
        'TF-IDF* + NN':'#900090',
        'GPT2 + NN':'#FFB000',
        'GPT2 + RUM?':'#FFB000',
        '===RUBINOVITZ===':'#00FFFF',
    }
    if model[1] in color:
        return color[model[1]]
    elif model[0] == '-empty':
        return '#505050'
    else:
        return '#909090'

def findfmt(model):
    n = len(model[1].split('/'))
    if n <= 2:
        if 'Tri' in model[1]:
            return '^'
        elif 'Quad' in model[1]:
            return 's'
        elif 'Quin' in model[1]:
            return 'p'
        else:
            return 'o'
    elif n == 3:
        return '^'
    elif n == 4:
        return 's'
    elif n == 5:
        return 'p'

def graph(infile="Comparison 200821.csv"):
    train = []
    val = []
    data = loadcsv(infile)
    bymodel = {}
    for line in data:
        if line[1] != 'Spacey' and line[1] != '-empty':
            continue
        model = (line[1], line[2])
        if model not in bymodel:
            bymodel[model] = ([], [])
        if line[9] == '' or line[10] == '':
            continue
        bymodel[model][0].append(float(line[9]))
        bymodel[model][1].append(float(line[10]))

    fig, ax = plt.subplots()
    fig.set_size_inches(11,8.5)
    ax.set_xlim(0,3)
    ax.set_ylim(0,1)
    for model in bymodel:
        print(model)
        # print(bymodel[model][0])
        # print(bymodel[model][1])
        if 'LINE' in model[1]:
            line = ax.plot(bymodel[model][0], bymodel[model][1], linestyle='-', c=findcolor(model), label=model[1])
        else:
            scatter = ax.scatter(bymodel[model][0], bymodel[model][1], marker=findfmt(model), c=findcolor(model), label=model[1], edgecolors='none')
    plt.show()
    # plt.save(infile[:-4]+".png")

if __name__ == "__main__":
    graph()
