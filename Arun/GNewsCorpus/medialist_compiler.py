import csv
from bs4 import BeautifulSoup
import requests

def loadcsv(filename):
    with open(filename, newline='') as f: 
        return list(csv.reader(f))

def savecsv(filename, list):
    with open(filename, "w", newline="") as f:
        csv.writer(f).writerows(list)

prefix = "https://news.google.com/search?q="
suffix = "&hl=en-US&gl=US&ceid=US:en"
def findtopicurl(topic):
    url = prefix + "%20".join(topic.split(" ")) + suffix
    print(url)
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')
    links = soup.find_all('a')
    hrefs = [link.get('href') for link in links]
    print(hrefs)
    hrefs = ["https://news.google.com" + x[1:] for x in hrefs if x is not None and x[:9] == "./topics/"]
    print(hrefs[0])
    return hrefs[0]

if __name__ == "__main__":
    data = loadcsv("Scrape_Topics/jiannatopics_named.csv")
    out = []
    for row in data:
        out.append([row[0], prefix + "%20".join(row[1].split(" ")) + suffix])
        #out.append([row[0], findtopicurl(row[1])])
    savecsv("medialistG.csv", out)