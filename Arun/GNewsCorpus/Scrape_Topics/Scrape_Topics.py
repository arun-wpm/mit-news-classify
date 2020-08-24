from bs4 import BeautifulSoup
from itntools import strip_url
from newspaper import Article
import csv
import requests
import pycountry
import time
import re
import pandas as pd

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

def filter_article_urls(tup, file='newsurlpatterns.csv'):
    common_blacklist = [
        r"^.*.png.*$",  # thecanary, motherjones
        r"^.*.jpg.*$",  # infowars
        r"^.*privacy-policy.*$",  # spectator, cbsnews
        r"^.*commons-community-guidelines$",  # commondreams
        r"^.*contact-us.*$",  # consortiumnews
        r"^https?://theconversation.com/institutions/.*$",  # theconversation
        r"^https?://theconversation.com/us/partners/.*$",  # theconversation
        r"^https?://theconversation.com/profiles/.*$",  # theconversation
        r"^(podcasts|itunes).apple.com/.*$",  # dailycaller
        # r"^.*facebook.com/.*$", #aljazeera, americanthinker, ap
        # r"^.*twitter.com/.*$", #aljazeera, ap
        # r"^.*snapchat.com/.*$", #nbcnews
        # r"^.*plus.google.com/.*$", #breitbart
        r"^mailto:.*$",  # ap
        r"^.*/aboutus/.*$",  # aljazeera
        r"^.*/terms-of-service/.*$",  # spectator
        r"^spectator.org/category/.*$",  # spectator
        r"^.*/about-breitbart-news.pdf$",  # breitbart
        # r"^coverageContainer/.*$", #cnn
        r"^.*/comment-policy/$",  # consortiumnews
        r"^.*/frequently-asked-questions$",  # economist
        r"^.*brand-use-policy$",  # eff
        r"^.*rss-feed.*$",  # fivethirtyeight
        r"^.*cookies-policy.*$",  # fivethirtyeight
        r"^.*career-opportunities.*$",  # foreignaffairs
        r"^.*my.*account.*$",  # foreignaffairs
        r"^.*foxnews.com/category/.*$",  # foxnews
        r"^.*mobile-and-tablet$",  # theguardian
        r"www.lewrockwell.com/books-resources/murray-n-rothbard-library-and-resources/",  # lewrockwell
        r"^.*podcast.*$",
        # nationalreview  # https://www.usatoday.com/story/entertainment/celebrities/2020/08/18/michelle-obama-brother-reveals-first-thoughts-barack-podcast/5584158002/
        r"^.*disable.*ad.*blocker.*$",  # slate
        r"^./category/.*$",
    ]

    newsurlpatterns = pd.read_csv(file)

    urls = tup[0]
    domain = tup[1]
    common_formula = newsurlpatterns[newsurlpatterns['url'].str.contains(domain)]['pattern']

    if len(common_formula) == 0 or common_formula.isna().any():
        common_formula = set(newsurlpatterns['pattern'])
        common_formula = {f for f in common_formula if pd.notna(f)}

    newsurls_df = pd.DataFrame(columns=['newsurl', 'newsrule'])
    nonnewsurls_df = pd.DataFrame(columns=['nonnewsurl', 'blackrule'])

    filtered = []
    for url in urls:
        if domain not in url:
            nonnewsurls_df = nonnewsurls_df.append({'nonnewsurl': url, 'blackrule': 'diff domain'}, ignore_index=True)
            continue
        blacklisted = False
        for formula in common_blacklist:
            if re.match(formula, url) is not None:
                blacklisted = True
                nonnewsurls_df = nonnewsurls_df.append({'nonnewsurl': url, 'blackrule': formula},
                                                       ignore_index=True)
                break
        if not blacklisted:
            commonformula = False

            for formula in common_formula:
                formula = formula[2:-1]
                if re.match(formula, url) is not None:
                    newsurls_df = newsurls_df.append({'newsurl': url, 'newsrule': formula}, ignore_index=True)
                    filtered.append(url)
                    commonformula = True
                    break
            if not commonformula:
                nonnewsurls_df = nonnewsurls_df.append({'nonnewsurl': url, 'blackrule': 'not in formula RIP'},
                                                       ignore_index=True)

    newsurls_df.sort_values(by=['newsrule', 'newsurl'], inplace=True)
    nonnewsurls_df.sort_values(by=['blackrule', 'nonnewsurl'], inplace=True)

    # newsurls_df.to_csv(domain.split(sep='.')[0] + '_newsurls.csv')
    # nonnewsurls_df.to_csv(domain.split(sep='.')[0] + '_nonnewsurls.csv')

    return filtered #, newsurls_df, nonnewsurls_df


def query_newsurl(url, domain=""):
    #strategy: finding all href=\\?\"[^\"]*\\?\"
    if (domain == ""):
        domain = "/".join(url.split('/')[:3]) + '/' # in the format of 'https://www.economist.com/'
    
    common_search = [r"href=\\?\"[^\" ]*\\?\"", r"href=\\?\'[^\' ]*\\?\'", r"\"uri\":\"[^\" ]*\"", r"\"url\":\"[^\" ]*\""]
    # search = "href"
    # print("Querying at " + url)

    # print(url)
    # page = requests.get(url)
    # soup = BeautifulSoup(page.text, 'html.parser')
    art = Article(url)
    art.download()
    soup = BeautifulSoup(art.html, 'html.parser')

    links = soup.find_all('a', href=True) + soup.find_all('div', href=True)  # added href = True
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
                hrefs.append(domain + href)

    # scripts = soup.find_all('script')  # TODO matches might be a repeat of links
    matches = []
    search = '|'.join(common_search)
    # matches.extend(re.findall(search, page.text))
    matches.extend(re.findall(search, art.html))

    # for script in scripts:
        # print(script.string)
        # print(re.search(search, str(script.string)))
        # matches.extend(re.findall(search, str(script.string)))
    for match in matches:
        # print("match:" + match)
        match = match.replace('\\\"', '\"')  # print('\\\"') leads to this: \"
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
                hrefs.append(domain + match)
    
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
