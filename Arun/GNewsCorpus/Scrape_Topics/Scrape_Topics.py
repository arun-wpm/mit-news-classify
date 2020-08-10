from bs4 import BeautifulSoup
from itntools import strip_url
import csv
import requests
import pycountry
import time
import re

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

def filter_article_urls(tup):
    urls = tup[0]
    domain = tup[1]
    common_formula = [
        r"^.*\/[0-9]{4}\/[0-9]{1,2}\/[0-9]{1,2}\/[0-9a-z.\/_?=&-]+$", #cnn, antiwar, breitbart, thecanary, cnbc, commondreams, consortiumnews, dailycaller, democracynow, economist, forbes
                                                                   #thegrayzone
        r"^.*\/(style|travel)\/article\/[0-9a-z.\/-]*$", #cnn
        r"^.*-[0-9]{15}.html$", #aljazeera
        r"^.*spectator.(org|co\.uk)/[0-9a-z\/]+-[0-9a-z\/-]{10,}$", #spectator
        r"^.*\/[0-9]{4}\/[0-9]{2}\/[0-9a-z.\/_?=-]{4,}$", #americanthinker, defenseone, eff, jacobinmag, lewrockwell, nationalreview
        r"^.*\/[0-9a-f]{32}$", #apnews
        r"^.*\/[0-9a-z-]*[0-9]{8,10}[a-z]*$", #bbc, businessinsider
        r"^.*\/[0-9a-z-]*[0-9]{4}-[0-9]{1,2}$", #businessinsider
        r"^.*\/[0-9a-z-]+-[0-9a-z-]{20,}\/?$", #businessinsider, buzzfeednews, fivethirtyeight, foreignaffairs, foxnews, theguardian, infowars, nbcnews (merge with spectator? too broad?)
        r"^.*\/[0-9a-z-]+-[0-9a-z-]{20,}\/[0-9]{7}$", #globalresearch
        r"^.*\/news\/[0-9a-z-]{10,}\/", #cbsnews
        r"^.*\/[0-9]{4}\/[0-9]{1,2}\/[0-9]{1,2}\/[0-9]{7}\/[0-9A-Za-z.\/-]+$", #dailykos (merge with cnn+?)
        r"^.*koreaherald.com/view.php\?ud=[0-9]{14}$", #koreaherald
        r"^.*\/[0-9a-z-]{10,}\/[0-9]{6}\/", #mintpressnews
        r"^.*\/[0-9]{4}\/[0-9]{4}\/c[0-9]{5}-[0-9]{7}.html$", #people.cn
        r"^.*\/article\/[0-9a-z\/-]+-id[0-9A-Z]{11}" #reuters
    ]
    common_blacklist = [
        r"^.*.png.*$", #thecanary, motherjones
        r"^.*.jpg.*$", #infowars
        r"^.*privacy-policy.*$", #spectator, cbsnews
        r"^.*commons-community-guidelines$", #commondreams
        r"^.*contact-us.*$", #consortiumnews
        r"^theconversation.com/institutions/.*$", #theconversation
        r"^theconversation.com/us/partners/.*$", #theconversation
        r"^theconversation.com/profiles/.*$", #theconversation
        r"^(podcasts|itunes).apple.com/.*$", #dailycaller
        # r"^.*facebook.com/.*$", #aljazeera, americanthinker, ap
        # r"^.*twitter.com/.*$", #aljazeera, ap
        # r"^.*snapchat.com/.*$", #nbcnews
        # r"^.*plus.google.com/.*$", #breitbart
        r"^mailto:.*$", #ap
        r"^.*/aboutus/.*$", #aljazeera
        r"^.*/terms-of-service/.*$", #spectator
        r"^spectator.org/category/.*$", #spectator
        r"^.*/about-breitbart-news.pdf$", #breitbart
        # r"^coverageContainer/.*$", #cnn
        r"^.*/comment-policy/$", #consortiumnews
        r"^.*/frequently-asked-questions$", #economist
        r"^.*brand-use-policy$", #eff
        r"^.*rss-feed.*$", #fivethirtyeight
        r"^.*cookies-policy.*$", #fivethirtyeight
        r"^.*career-opportunities.*$", #foreignaffairs
        r"^.*my.*account.*$", #foreignaffairs
        r"^.*foxnews.com/category/.*$", #foxnews
        r"^.*mobile-and-tablet$", #theguardian
        r"www.lewrockwell.com/books-resources/murray-n-rothbard-library-and-resources/", #lewrockwell
        r"^.*podcast.*$", #nationalreview
        r"^.*disable.*ad.*blocker.*$", #slate
        r"^./category/.*$",
    ]
    filtered = []
    for url in urls:
        if domain not in url:
            continue
        blacklisted = False
        for formula in common_blacklist:
            if re.match(formula, url) is not None:
                blacklisted = True
                break
        if not blacklisted:
            for formula in common_formula:
                if re.match(formula, url) is not None:
                    filtered.append(url)
                    break
    return filtered

def query_newsurl(url, domain=""):
    #strategy: finding all href=\\?\"[^\"]*\\?\"
    if (domain == ""):
        domain = "/".join(url.split('/')[:3]) + '/'
    
    common_search = [r"href=\\?\"[^\" ]*\\?\"", r"href=\\?\'[^\' ]*\\?\'", r"\"uri\":\"[^\" ]*\"", r"\"url\":\"[^\" ]*\""]
    # search = "href"
    # print("Querying at " + url)

    # print(url)
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')
    links = soup.find_all('a') + soup.find_all('div')
    links = [link.get('href') for link in links]
    hrefs = []
    for href in links:
        if (href is not None):
            if (href[:7] == 'http://'):
                # print(href[7:])
                hrefs.append(href)
            elif (href[:8] == 'https://'):
                # print(href[8:])
                hrefs.append(href)
            elif (href[:2] == '//'):
                # print(href[2:])
                hrefs.append(href[2:])
            elif (href[:1] == '/'):
                # print(href[1:])
                hrefs.append(domain + href[1:])
            else:
                # print(href)
                hrefs.append(href)

    # scripts = soup.find_all('script')
    matches = []
    search = '|'.join(common_search)
    matches.extend(re.findall(search, page.text))
    # for script in scripts:
        # print(script.string)
        # print(re.search(search, str(script.string)))
        # matches.extend(re.findall(search, str(script.string)))
    for match in matches:
        # print("match:" + match)
        match = match.replace('\\\"', '\"')
        match = match.replace('\\/', '/')
        # print(match)
        match = re.split(r'\"|\'', match)[-2]
        # print(match)
        if (match is not None):
            if (match[:7] == 'http://'):
                # print(match[7:])
                hrefs.append(match)
            elif (match[:8] == 'https://'):
                # print(match[8:])
                hrefs.append(match)
            elif (match[:2] == '//'):
                # print(match[2:])
                hrefs.append(match[2:])
            elif (match[:1] == '/'):
                # print(match[1:])
                hrefs.append(domain + match[1:])
            else:
                # print(match)
                hrefs.append(match)
    
    hrefs = list(dict.fromkeys(hrefs)) # remove duplicates

    return hrefs, strip_url(url)

def query_gnewsurl(url):
    print("Querying at " + url)

    print(url)
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')
    links = soup.find_all('a')
    hrefs = [link.get('href') for link in links]
    hrefs = ["https://news.google.com" + x[1:] for x in hrefs if x is not None and x[:11] == "./articles/"]
    hrefs = list(dict.fromkeys(hrefs)) # remove duplicates
    return hrefs

    # solve redirects
    
    # articles = []
    # for href in hrefs:
    #     # find the actual article url after redirect
    #     redirect = requests.get(href)
    #     print(redirect, redirect.url)
    #     articles.append(redirect.url)
    #     # print("stalling...")
    #     # time.sleep(WAIT)
    
    # # each two are duplicates, and the last one is useless
    # print("Take a look!")
    # # for url in articles:
    #     # print(">> " + url[0])
    # return articles

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
