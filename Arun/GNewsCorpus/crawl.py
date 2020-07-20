# USAGE EXAMPLE: python crawl.py medialist1.csv >& qaz_crawl.log
# Max Tegmark, March 2020
# Continuously crawls the web, both scraping & downloading articles
# from a given media list leaving enough time between repeats.
# For each media outlet, keeps time of last crawl, time of last scrape & 
# list of URLs to be-downloaded. 

import newspaper, csv, sys, os, time, datetime, re
import numpy as np
from newspaper import Article
from useful import *
from itntools import *
from Scrape_Topics import query_gnewsurl

# Return time in seconds since startTime:
def Time(): return (datetime.datetime.now() - baseTime).total_seconds() 

def reset_cache(media_url):
    paper = newspaper.build(media_url)
    paper.clean_memo_cache()
    print("Cleared cache for",media_url)

def reset_caches(mediafile):
    media = loadcsv(mediafile)
    print("Will clear cache for",len(media),"media sites.")
    for m in media: reset_cache(m[1])
    print("ALL DONE")
    
def url2article(url):
    scrapetime = datetime.datetime.now().isoformat()[:19]
    if url_valid(url):
        try:
            a = Article(url)
            a.download()
            a.parse()
            a.nlp()
            try:
                pubtime = a.publish_date.isoformat()[:19]
            except:
               try:
                  pubtime = pubtime.isoformat()[:19]
               except:
                   pubtime = ""
            result = [url,pubtime,a.authors,a.title,a.keywords,a.summary,a.text,a.top_image,scrapetime]
        except:
            print("Download failed for ",url)
            result = [url,'','','DOWNLOAD FAILED','','','',"",scrapetime]
    else:
        print("Invalid URL: ",url)
        result = [url,'','','INVALID URL','','','',scrapetime]
    return result

# Scrape url & append resulting article to appropriate file determined by URL:
def scrape(media_name,url):
    print("Scraping",url,"from",media_name,"...")
    articlefile = "articles_"+media_name+"_crawled.tsv"
    article = url2article(url)
    print("### appending article with url",article[0],"to",articlefile)
    appendtsv(articlefile,[article])

# def crawl(media_url):	
#     print("Crawling",media_url,"...")
#     paper = newspaper.build(media_url)
#     return list([article.url for article in paper.articles])

def crawl(media_url):
    # from each topic in Google News in medialistG.csv, get a list of articles (ordered)
    print("Crawling", media_url, "...")
    return query_gnewsurl(media_url)

#def scrape(media_name,url):
#    print("Fake-scraping",url)
#    time.sleep(0.5)

#def crawl(media_url):	
#    print("Fake-crawling",media_url)
#    time.sleep(.5)
#    return ["cnn.com/corinavirus.html","futureoflife.com/krakovnaSexScandal.html/"]

def nexttime(s):
    urls_to_scrape = s[4:]
    if len(urls_to_scrape) == 0:
        return s[2]+crawlWait
    else: 
        return min(s[2]+crawlWait,s[3]+scrapeWait)
    
def act(i):
    global schedule, lastUpdateTime, totWait, totDelay, updates, crawls, scrapes
    (media_name,media_url,lastCrawlTime,lastScrapeTime) = schedule[i][:4]
    urls_to_scrape = schedule[i][4:]
    if Time() > lastUpdateTime + updateWait: # Update
        lastUpdateTime = Time()
        updates += 1
        savecsv(schedulefile,schedule)
        if crawls+scrapes > 0:
            (avgWait,avgDelay) = (totWait/(crawls+scrapes),totDelay/(crawls+scrapes))
            print("### After",(Time()-startTime)/60,"minutes, done",crawls,"crawls &",scrapes,"scrapes with average wait",avgWait,"& delay",avgDelay) 
    delay = Time()-(lastCrawlTime + crawlWait)
    if delay > 0:  # Crawl
        totDelay = totDelay + delay
        lastCrawlTime = Time()
        crawls += 1
        urls = crawl(media_url)
        print("   ",len(urls),"URLs found.")
        #savecsv(schedulefile,schedule)
        urls_to_scrape = urls_to_scrape + urls
    delay = Time()-(lastScrapeTime + scrapeWait)
    #print("### Scrape?",urls_to_scrape)
    if delay>0 and len(urls_to_scrape)>0:  # Scrape
        totDelay = totDelay + delay
        lastScrapeTime = Time()
        scrapes += 1
        url = urls_to_scrape.pop(0)
        scrape(media_name,url)
    schedule[i] = [media_name,media_url,lastCrawlTime,lastScrapeTime]+urls_to_scrape
    savecsv(schedulefile,schedule) ###
    return

# SET PARS: 
mediafile = sys.argv[1]
#mediafile = "medialist.csv"
#mediafile = "medialist_short.csv"
schedulefile = "scheduleG.csv"
#(updateWait,crawlWait,scrapeWait) = (10,10*60,4)
(updateWait,crawlWait,scrapeWait) = (60,5*60,4)
(updates,crawls,scrapes,totWait,totDelay) = (0,0,0,0.,0.)

# INITIALIZE:
media = loadcsv(mediafile)
print("Loaded",len(media),"media sites to repeatedly crawl.")
baseTime = datetime.datetime.fromisoformat("2020-01-01T00:00:00.000000")                                  
startTime = Time()

# INITIALIZE SCHEDULE:
# Each row will contain (media_name,media_url,lastCrawlTime,lastScrapeTime,urls_to_scrape)
now = Time()
lastUpdateTime = now-updateWait
schedule = robustloadcsv(schedulefile)
for i in range(len(schedule)):
    schedule[i][2] = float(schedule[i][2])
    schedule[i][3] = float(schedule[i][3])
if len(schedule) != len(media):
    print("Creating new schedule from scratch.")
    schedule = [[m[0],m[1],now-crawlWait,now-scrapeWait] for m in media]
    savecsv(schedulefile,schedule)
  
nloops = 10 # Don't go forever, since there seems to be a memory leak somewhere  
for loop in range(nloops):
    print("##### Starting loop",loop)
    for i in range(len(schedule)): act(i)
    next = min(list(map(nexttime,schedule)))
    #print("### Time:",Time()-startTime)
    #print("### Next:",next-startTime,lastUpdateTime+updateWait-startTime,list(map(nexttime,schedule)))
    next = min(next,lastUpdateTime+updateWait)
    wait = next-Time()
    #print("### Wait:",wait)
    if wait > 0:
        print("Taking a",wait,"second nap...")
        time.sleep(wait)
        totWait += wait
print("##### Quitting & restarting...")

# reset_caches("medialist3a.csv")

