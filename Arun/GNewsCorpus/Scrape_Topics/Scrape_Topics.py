from bs4 import BeautifulSoup
from itntools import strip_url
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

def filter_article_urls(tup):
    urls = tup[0]
    domain = tup[1]
    common_formula = [
        # TODO 2 dashes
        # https://www.cnn.com/2020/06/16/politics/cia-wikileaks-vault-7-leak-report/index.html
        r"^.*\/[0-9]{4}\/[0-9]{1,2}\/[0-9]{1,2}\/[0-9a-z.\/_?=&-]+$", #cnn, antiwar, breitbart, thecanary, cnbc, commondreams, consortiumnews, dailycaller, democracynow, economist, forbes
                                                                   #thegrayzone, currentaffairs
        # TODO 2 dashes
        r"^.*\/(style|travel)\/article\/[0-9a-z.\/-]*$", #cnn

        # TODO 2 dashes
        # https://www.aljazeera.com/news/2020/08/steve-bannon-trump-adviser-arrested-fraud-200820134920664.html
        r"^.*-[0-9]{15}.html$", #aljazeera

        # https://spectator.org/day-three-of-the-democrat-pros-and-cons-as-pr-pros-combine-to-con-america/
        r"^.*spectator.(org|co\.uk)/[0-9a-z\/]+-[0-9a-z\/-]{10,}$", #spectator

        # https://jacobinmag.com/2020/08/medicare-for-all-for-profit-health-democratsq
        r"^.*\/[0-9]{4}\/[0-9]{2}\/[0-9a-z.\/_?=-]{4,}$", #americanthinker, defenseone, eff, jacobinmag, lewrockwell, nationalreview


        r"^.*\/[0-9a-f]{32}$", #apnews

        #
        r"^.*\/[0-9a-z-]*[0-9]{8,10}[a-z]*$", #bbc, businessinsider

        # https://www.businessinsider.com/tesla-dating-app-owners-only-car-obsessed-elon-musk-stans-2020-8
        r"^.*\/[0-9a-z-]*[0-9]{4}-[0-9]{1,2}$", #businessinsider


        r"^.*\/[0-9a-z-]+-[0-9a-z-]{20,}\/?$", #businessinsider, buzzfeednews, fivethirtyeight, foreignaffairs, foxnews, theguardian, infowars, nbcnews (merge with spectator? too broad?)


        r"^.*\/[0-9a-z-]+-[0-9a-z-]{20,}\/[0-9]{7}$", #globalresearch


        r"^.*\/news\/[0-9a-zA-z-]{12,}\/?", #cbsnews, changed from 10 to 12


        r"^.*\/[0-9]{4}\/[0-9]{1,2}\/[0-9]{1,2}\/[0-9]{7}\/[0-9A-Za-z.\/-]+$", #dailykos (merge with cnn+?)


        r"^.*koreaherald.com/view.php\?ud=[0-9]{14}$", #koreaherald

        # https://www.mintpressnews.com/disturbing-milestone-top-12-plutocrats-hold-1-trillion-wealth-inequality/270569/
        r"^.*\/[0-9a-z-]{10,}\/[0-9]{6}\/", #mintpressnews

        # http://pic.people.com.cn/n1/2020/0820/c1016-31830437.html
        r"^.*\/[0-9]{4}\/[0-9]{4}\/c[0-9]{5}-[0-9]{7}.html$", #people.cn

        # https://www.reuters.com/article/us-yemen-security-explainer/why-yemen-is-at-war-idUSKCN22924D
        r"^.*\/article\/[0-9a-z\/-]+-id[0-9A-Z]{11}",  #reuters

        # https://www.dailymail.co.uk/femail/article-8613509/The-best-tennis-gear-step-game-summer.html
        r"^.*\/article-\d*\/[0-9a-zA-Z-]{12,}\/?"  # dailymail
        # r"^.*\/article-\d*\/[0-9a-zA-Z-]{12,}.html?"  # dailymail
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
        r"^.*podcast.*$", #nationalreview  # https://www.usatoday.com/story/entertainment/celebrities/2020/08/18/michelle-obama-brother-reveals-first-thoughts-barack-podcast/5584158002/
        r"^.*disable.*ad.*blocker.*$", #slate
        r"^./category/.*$",
    ]

    print('filtering out life')
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
                if re.match(formula, url) is not None:
                    newsurls_df = newsurls_df.append({'newsurl': url, 'newsrule': formula}, ignore_index=True)
                    filtered.append(url)
                    commonformula = True
                    break
            if not commonformula:
                nonnewsurls_df = nonnewsurls_df.append({'nonnewsurl': url, 'blackrule': 'not in formula RIP'}, ignore_index=True)

    newsurls_df.sort_values(by=['newsrule', 'newsurl'], inplace=True)
    nonnewsurls_df.sort_values(by=['blackrule', 'nonnewsurl'], inplace=True)

    newsurls_df.to_csv(domain.split(sep='.')[0] + '_newsurls.csv')
    nonnewsurls_df.to_csv(domain.split(sep='.')[0] + '_nonnewsurls.csv')

    return filtered


from urllib.parse import urlparse
def rwfiletered(tup):
    urls = tup[0]
    domain = tup[1]

    newsurls_df = pd.DataFrame(columns=['newsurl', 'newsrule'])
    nonnewsurls_df = pd.DataFrame(columns=['nonnewsurl', 'blackrule'])

    for url in urls:
        parsedurl = urlparse(url)
        if domain in parsedurl.netloc:
            splitpath = (parsedurl.path).split(sep='/')

            isnews = False
            # note that this doesn't work for ap news https://apnews.com/9dd9d8dbff298d28e396ec0983adaefd or https://www.bbc.com/sport/basketball/53677658
            if re.match(r"([A-Za-z0-9]+-[A-Za-z0-9]+-[A-Za-z0-9]+)", splitpath[-1]):
                print(-1, url)
                isnews = True
                newsurls_df = newsurls_df.append({'newsurl': url, 'newsrule': '-1 enough dashes'}, ignore_index=True)
            elif len(splitpath) >= 2 and re.match(r"([A-Za-z0-9]+-[A-Za-z0-9]+-[A-Za-z0-9]+)", splitpath[-2]):
                print(-2, url)
                isnews = True
                newsurls_df = newsurls_df.append({'newsurl': url, 'newsrule': '-2 enough dashes'}, ignore_index=True)

            if not isnews:
                print('\tRIP', url)
                nonnewsurls_df = nonnewsurls_df.append({'nonnewsurl': url, 'blackrule': 'not enough dashes'},
                                                       ignore_index=True)

    return newsurls_df, nonnewsurls_df






def query_newsurl(url, domain=""):
    #strategy: finding all href=\\?\"[^\"]*\\?\"
    if (domain == ""):
        domain = "/".join(url.split('/')[:3]) + '/' # in the format of 'https://www.economist.com/'
    
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
