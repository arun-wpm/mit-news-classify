# import pickle
import csv

topicfile = "nyt-theme-tags.csv"


def get_mapping(topicfile):
    data = loadcsv(topicfile)
    id_mapping = {int(t[1]): int(t[0]) for t in data[::2][1:]}
    topic_mapping = {int(t[0]): t[2] for t in data[::2][1:]}
    return id_mapping, topic_mapping


def long_to_short_topic_ids(topics, id_mapping):
    new_topics = []
    for topic in topics:
        if topic in id_mapping:
            new_topics.append(id_mapping[topic])
    return new_topics


def loadcsv(filename):
    with open(filename, newline='') as f:
        return list(csv.reader(f))


def flatten(listoflists):
    return list([j for i in listoflists for j in i])


def str2int(lst):
    return list(map(int, lst))


# id_mapping, topic_mapping = get_mapping(topicfile)
# print(id_mapping)
# print(topic_mapping)

# infile = "../../nyt_corpus/NYTcorpus.p"
#
# print("Loading articles from", infile, "...")
# articles = pickle.load(open(infile, "rb"))
#
# NYTtagset = set(str2int(flatten([a[3:] for a in articles])))

# # LOAD TOPIC DEFINITIONS:
# topics = loadcsv(topicfile)
# # validate_topics(topics)outfile = "jiannatags.csv"
# topics = loadcsv(topicfile)
# tagdata = [t for t in topics[1:] if len(t[1]) > 4]

# print("What is tagdata??:", tagdata[5])
# print(tagdata[23])
#
# topicdict = {int(t[1]): t[0] for t in tagdata}
# count = 0
# tagset = set([int(t[1]) for t in tagdata])
# unknown = [t for t in NYTtagset if topicdict.get(t, "") == ""]
# print(len(unknown), "of", len(NYTtagset), "tags are missing from topics.csv")
