from bs4 import BeautifulSoup
import csv
import requests
import pycountry
import time

WAIT = 5 # stall time so google doesn't block me ;-;

def loadcsv(filename):
    with open(filename, newline='') as f: 
        return list(csv.reader(f))

def savecsv(filename, list):
    with open(filename, "w", newline="") as f:
        csv.writer(f).writerows(list)

def get_urls(txt):
    print(txt)
    soup = BeautifulSoup(txt.text, 'html.parser')
    links = soup.find_all('a')
    articles = []
    for link in links:
        href = link.get('href')
        if href[:5] == "/url?": # filter to only those which are links to outside google
            params = [tuple(x.split('=')) for x in href[5:].split('&')]
            # print(params)
            # we only need the q parameter, changeable
            articles.append([params[0][1]])
    # each two are duplicates, and the last one is useless
    return articles[:-1:2]

def craft(base, params):
    base = base
    first = True
    for (x, y) in params:
        base = base + ("?" if first else "&") + x + "=" + y.lower().replace(' ', '+').replace('_', '+')
        first = False
    return base

gurl = "https://www.google.com/search" # "https://www.google.com/search?q=thailand&tbm=nws&start=10"
gnewsurl = "https://news.google.com/search" # "https://news.google.com/search?q=Thailand&hl=en-US&gl=US&ceid=US%3Aen"
def query_topic(topic="thailand", n=10):
    print("Querying for " + topic + ", n=" + str(n))
    articles = []
    for i in range(0, n, 10):
        url = craft(gurl, [("q", topic), ("tbm", "nws"), ("start", str(i))])
        print(url)
        page = requests.get(url)
        articles.extend(get_urls(page))
        print("stalling...")
        time.sleep(WAIT)
    # print("Take a look!")
    # for url in articles:
        # print(">> " + url[0])
    return articles

def query_gurl(url):
    print("Querying at " + url)
    articles = []
    for i in range(0, 1000, 10):
        print(url)
        page = requests.get(url + ("&start=" + str(i) if i > 0 else ""))
        results = get_urls(page)
        if len(results) == 0:
            break
        articles.extend(results)
        print("stalling...")
        time.sleep(WAIT)
    print("Take a look!")
    for url in articles:
        print(">> " + url[0])
    return articles

def query_gnewsurl(url):
    print("Querying at " + url)

    print(url)
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')
    links = soup.find_all('a')
    hrefs = [link.get('href') for link in links]
    hrefs = ["https://news.google.com" + x[1:] for x in hrefs if x is not None and x[:11] == "./articles/"]
    hrefs = list(dict.fromkeys(hrefs)) # remove duplicates
    articles = []
    for href in hrefs:
        # find the actual article url after redirect
        redirect = requests.get(href)
        print(redirect, redirect.url)
        articles.append(redirect.url)
        # print("stalling...")
        # time.sleep(WAIT)
    
    # each two are duplicates, and the last one is useless
    print("Take a look!")
    # for url in articles:
        # print(">> " + url[0])
    return articles

def query_all(qlist, n=10):
    print("Start the scraping!")
    for topic in qlist:
        if isinstance(topic, tuple):
            articles = query_topic(topic[1], n)
            savecsv(topic[0] + ".csv", articles)

        elif isinstance(topic[0], tuple):
            articles = query_topic(topic[0][1], n)
            savecsv(topic[0][0] + ".csv", articles)
        else:
            articles = query_topic(topic[0], n)
            savecsv(topic[0] + ".csv", articles)
        print("more stalling...")
        time.sleep(WAIT)

def convert_country_names(qlist):
    print("Converting country codes to names...")
    for i in range(len(qlist)):
        code = qlist[i][0]
        if len(code) > 3 or not code.isupper():
            continue
        try:
            country = pycountry.countries.get(alpha_3=code)
            qlist[i][0] = (code, country.name)
        except Exception:
            continue
    return qlist


if __name__ == "__main__":
    qlist2 = loadcsv("../../../jiannatopics_named.csv")

    # qlist = loadcsv("../../jiannatopics.csv")
    # qlist = convert_country_names(qlist)

    query_all(qlist2, 1000)
