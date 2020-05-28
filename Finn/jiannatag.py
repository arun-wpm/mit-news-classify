# USAGE EXAMPLE: python3 jiannatag.py
# By Jianna Liu with some tweaks by Max Tegmark, March-April 2020
import csv
import re
import sys
import pickle
# from useful import *


def loadcsv(filename):
    with open(filename, newline='') as f:
        return list(csv.reader(f))


def savecsv(filename, list):
    with open(filename, "w", newline="") as f:
        csv.writer(f).writerows(list)


def flatten(listoflists):
    return list([j for i in listoflists for j in i])


def str2int(lst):
    return list(map(int, lst))


def split(string):
    '''
    Method splits country search string such as "samoa NOT joe"
    into an array that contains ['samoa', 'NOT', 'joe]

    Takes in country search string as input and outputs it as an array
    '''
    result = []
    while len(string) > 0:
        if string[0] == '(':
            result.append('(')
            string = string[1:]
        elif string[0] == '"':
            end_quote = string[1:].index('"') + 1
            result.append(string[1:end_quote])
            string = string[end_quote + 1:]
        elif string[0] == ' ':
            string = string[1:]
        else:
            if ' ' in string:
                space_ind = string.index(' ')
                result.append(string[:space_ind])
                string = string[space_ind + 1:]
            else:
                result.append(string)
                string = ''
    return result


def deconstruct_string(country,searchString):
    '''
    Method seeks to turn original country search into a string of boolean
    variables.

    For example, if we are searching British OR Virgin, this will then change
    this string to True OR False, depending on whether "British" is in the article
    and if "Virgin" is in the article.

    Takes a country string and the article to search through as input, and outputs
    an altered string of boolean variables.
    '''
    new_country = split(country)
    result = ''
    for string in new_country:
        if not (string == 'AND' or string == 'and' or string == 'OR' or string == 'or' or string == 'NOT' or string == 'not'):
            endParen = False
            if string[0] == '(':
                result += '('
                string = string[1:]
                continue
            elif (len(string) > 1 and string[-1] == ')'):
                string = string[:-1]
                endParen = True
            elif string[0] == ')':
                result += ')'
                string = string[1:]
                continue
            in_article = (string in searchString)
            result += str(in_article)
            if endParen:
                result += ')'
        elif string == 'AND' or string == 'and':
            result += ' and '
        elif string == 'OR' or string == 'or':
            result += ' or '
        elif string == "NOT" or string == 'not':
            result += ' not '
    return result


def determine_expression(alt_string):
    '''
    Method seeks to convert the newly constructed string of booleans
    to a simple boolean

    For example, True OR False will simplify to True

    Has a string as an input and outputs a boolean
    '''
    if len(alt_string) <= 5:
        return alt_string
    else:
        if '(' in alt_string:
            beg_paren = alt_string.index('(')
            end_paren = alt_string.index(')')
            evaluate = alt_string[beg_paren + 1: end_paren]
            new_string = determine_expression(evaluate)
            return determine_expression(alt_string[:beg_paren] + new_string)
        if 'not' in alt_string:
            not_index = alt_string.index('not')
            alt_string = alt_string[:not_index] + \
                'and' + alt_string[not_index + 3:]
            not_index += 4
            if alt_string[not_index] == "F":  # if false turn to true
                alt_string = alt_string[:not_index] + \
                    "True" + alt_string[not_index + 5:]
            elif alt_string[not_index] == "T":
                alt_string = alt_string[:not_index] + \
                    "False" + alt_string[not_index + 4:]
            new_string = determine_expression(alt_string)
            return determine_expression(alt_string[:not_index] + new_string)
        if 'and' in alt_string:
            begin_and = alt_string.index('and')
            if alt_string[begin_and - 3] == 'u' and alt_string[begin_and + 4] == 'T':
                alt_string = alt_string[0: begin_and -
                                        5] + 'True' + alt_string[begin_and+8:]
            else:
                if alt_string[begin_and - 3] == 's' and alt_string[begin_and + 4] == 'T':
                    alt_string = alt_string[0: begin_and - 6] + \
                        'False' + alt_string[begin_and+8:]
                elif alt_string[begin_and - 3] == 'u' and alt_string[begin_and + 4] == 'F':
                    alt_string = alt_string[0: begin_and - 5] + \
                        'False' + alt_string[begin_and+9:]
                else:
                    alt_string = alt_string[0: begin_and - 6] + \
                        'False' + alt_string[begin_and+9:]
            return determine_expression(alt_string)
        if 'or' in alt_string:
            begin_or = alt_string.index('or')
            if alt_string[begin_or - 3] == 'u' and alt_string[begin_or + 3] == 'T':
                alt_string = alt_string[0: begin_or -
                                        5] + 'True' + alt_string[begin_or+7:]
            else:
                if alt_string[begin_or - 3] == 'u' and alt_string[begin_or + 3] == 'F':
                    alt_string = alt_string[0: begin_or - 5] + \
                        'True' + alt_string[begin_or+8:]
                elif alt_string[begin_or - 3] == 's' and alt_string[begin_or + 3] == 'T':
                    alt_string = alt_string[0: begin_or - 6] + \
                        'True' + alt_string[begin_or+7:]
                else:
                    alt_string = alt_string[0: begin_or -
                                            6] + 'False' + alt_string[begin_or+8:]
            return determine_expression(alt_string)


def listCountries(countryinfo, text):
    '''
    Method takes in title and body of article and list of the original country name,
    its three letter code, and the string we are searching for.
    '''
    (acronyms, countries, search_strings) = countryinfo
    result = []
    for i in range(len(countries)):
        # print(countries[i] in text, 'regular country in text')
        # print(countries[i].lower() in text, 'lowercased country in text')
        if countries[i].lower() in text and acronyms[i] not in result and countries[i] != '':
            if countries[i].lower() == search_strings[i].replace('"', '').lower():
                # print('in if loop')
                result.append(acronyms[i])
            else:
                # print('in else loop')
                country = deconstruct_string(search_strings[i], text)
                in_article = determine_expression(country)
                if in_article:
                    result.append(acronyms[i])
# Let's instead explicitly add US states and large cities, as well as "London", etc
#    if result == []:  # because usually discussing states in USA
#        result = ['USA']
    return result


def unpackTags(topicdict, taglist):
    return [topicdict[t] for t in taglist if t in tagset]


def sstr(t):
    if len(t) < 19:
        return ""
    else:
        return t[18]


def score(tags1, tags2):
    correct = [t for t in tags2 if t in tags1]
    falseNegatives = [t for t in tags1 if t not in tags2]
    falsePositives = [t for t in tags2 if t not in tags1]
    return (correct, falseNegatives, falsePositives)


def tags_dict():
    with open('/home/euler/DATA1/nyt_corpus/nyt-theme-tags.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)

    csv_dict = {}
    for tagList in data:
        if len(tagList) > 1 and tagList[0] != 'tags_id':
            csv_dict[tagList[0]] = tagList[1]
    return csv_dict


infile = "NYTcorpus.p"
# infile = "NYTcorpus_sample.p"
outfile = "jiannatags.csv"
topicfile = "topics.csv"

# LOAD TAGGED NYT DATA:
print("Loading articles from", infile, "...")
articles = pickle.load(open(infile, "rb"))
# pickle.dump(articles[:10000],open("NYTcorpus_sample.p","wb"))
NYTtagset = set(str2int(flatten([a[3:] for a in articles])))
csv_dict = tags_dict()

# LOAD TOPIC DEFINITIONS:
topics = loadcsv(topicfile)
# validate_topics(topics)outfile = "jiannatags.csv"
topicfile = "topics.csv"
topics = loadcsv(topicfile)
tagdata = [t for t in topics[1:] if len(t[1]) > 4]
topicdict = {int(t[1]): t[0] for t in tagdata}
count = 0
tagset = set([int(t[1]) for t in tagdata])
unknown = [t for t in NYTtagset if topicdict.get(t, "") == ""]
print(len(unknown), "of", len(NYTtagset), "tags are missing from topics.csv and will be ignored")

# TAG ARTICLES:
acronyms = str2int(list([t[1] for t in tagdata]))
tags = list([t[4] for t in tagdata])
search_strings = list([sstr(t) for t in tagdata])
taginfo = (acronyms, tags, search_strings)
print("Tagging", len(articles), "articles from", infile, "...")
JiannaTags = []
count = 0
total = 0
for a in articles:
#    print(count, 'count')
#    if count == 9:
#        print(a[1]+a[2], 'supposed text with lawsuit')
#    if count == 13:
#        print(a[1]+a[2], 'supposed text with hostags')
#        print('hostages' in a[1]+a[2], 'Hostages'.lower() in a[1]+a[2])
#    if count == 17:
#        print(a[1]+a[2], 'supposed text with judges')
    JiannaTags.append(unpackTags(topicdict, listCountries(taginfo, a[1]+a[2])))
    if JiannaTags[-1] != []:
        count += 1
    total += 1
#    print(JiannaTags, 'jiannatags')
#    if count == 22:
#        break
# JiannaTags = [unpackTags(topicdict,listCountries(taginfo,a[1]+a[2])) for a in articles]
# JiannaTags = [unpackTags(topicdict,listCountries(taginfo,a[1]+a[2])) for a in articles[:1000]]
print(count/total, 'count/total')
savecsv(outfile, JiannaTags)
print("Jianna-tags saved in", outfile)

# ANALYZE ACCURACY:
print("Analyzing tagging accuracy...")
NYTtags = [unpackTags(topicdict, str2int(a[3:])) for a in articles]
scores = [score(NYTtags[i], JiannaTags[i]) for i in range(len(JiannaTags))]
savecsv("tag_comparison.csv", [[len(s[0]), len(s[1]), len(s[2])]+s[0]+s[1]+s[2] for s in scores])
n = len(scores)
correct = sum([len(s[0]) for s in scores])
falseNegatives = sum([len(s[1]) for s in scores])
falsePositives = sum([len(s[2]) for s in scores])
print("Correct tags/articls........", correct/n)
print("False negatives/article.....", falseNegatives/n)
print("False positves/article......", falsePositives/n)
