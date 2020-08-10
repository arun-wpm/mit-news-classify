import re
import matplotlib.pyplot as plt
import math

def graph(infile="nn_17573234.out", show=False, log=False):
    train = []
    val = []
    with open(infile, "r") as f:
        while True:
            line = f.readline()
            if len(line) == 0:
                break
            match = re.findall(r"loss: [0-9.]+ - binary_crossentropy: [0-9.]+ - val_loss: [0-9.]+ - val_binary_crossentropy: [0-9.]+$", line)
            # print(match)
            if len(match) > 0:
                print(match[0])
                match = match[0]
                match = re.split(r"( - |: )", match)
                # print(match)
                train.append(math.log10(float(match[2])) if log else float(match[2]))
                val.append(math.log10(float(match[10])) if log else float(match[10]))

    print(train)
    print(val)
    if show:
        x = [i for i in range(len(train))]
        fig, ax = plt.subplots()
        line1 = ax.plot(x, train, label="Train")
        line2 = ax.plot(x, val, label="Validation")
        ax.legend(loc='upper right')
        plt.show()
    # plt.save(infile[:-4]+".png")
    return train, val

if __name__ == "__main__":
    all_train = []
    all_val = []
    x = [i+1 for i in range(50)]
    legend=["without GPT2", "without doc2vec", "without TF-IDF bigrams", "without TF-IDF"]
    fig, ax = plt.subplots()
    for i in range(4, 8):
        train, val = graph("nn_1757323"+str(i)+".out")
        ax.plot(x, val, label=legend[i-4])
        ax.legend(loc='upper right')
    plt.show()
