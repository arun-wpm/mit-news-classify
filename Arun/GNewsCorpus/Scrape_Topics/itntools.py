# Shared tools for the ITN project
# Max Tegmark, March 2020

import re, datetime, pytz
import numpy as np
from useful import *

# For url-matching, replace "https" by "http" and drop anything after "?" or "#":
def simplify_url(url):
    u = url.replace("https://","http://")
    try: 
        i = u.index("?")
        u = u[:i]
    except:
        pass
    try: 
        i = u.index("#")
        u = u[:i]
    except:
        pass
    #if "reuters.com" in u: u="reuters.com" # Deal with "fr.reuters.com", etc
    return u

# Remove duplicates rows that have the same url in column number <urlcol>.
# Matches simplified url's and keeps whichever row has shortest version.
#def remove_duplicates(table,urlcol):
#    urls = np.array([[i,table[i][urlcol]] for i in range(len(table))])
#    urls[:,1] = list(map(simplify_url,urls[:,1]))
#    urls = mysort(urls,1)
#    # Keep only ones different from their successor, and the last one:
#    OK = [urls[i,1] != urls[i+1,1] for i in range(len(urls)-1)]+[True]
#    idx = np.arange(len(OK))[OK]
#    print(len(table)-len(idx),"duplicates deleted, leaving",len(idx),"rows")
#    return [table[int(urls[i,0])] for i in idx]

# Remove duplicates rows that have the same url in column number <urlcol>.
# Matches simplified url's and keeps whichever row has oldest version.
def remove_duplicates(table,urlcol,timecol):
    if len(table) == 0: return table
    iut = [[i,simplify_url(table[i][urlcol]),table[i][timecol]] for i in range(len(table))]
    iut = sorted(iut, key=lambda x: (x[1], -float(x[2])))
    # Keep only ones different from their successor, and the last one:
    OK = [iut[i][1] != iut[i+1][1] for i in range(len(iut)-1)]+[True]
    idx = np.arange(len(OK))[OK]
    print(len(table)-len(idx),"URL-duplicates deleted, leaving",len(idx),"rows")
    keep = [int(iut[i][0]) for i in idx]
    keep.sort()
    return [table[k] for k in keep]

# Remove duplicates rows that have the same mediaID and title in column numbers <mediacol> and <titlecol>.
# Matches simplified url's and keeps whichever row has oldest version.
def remove_duplicates2(table,mediacol,titlecol,timecol):
    if len(table) == 0: return table
    imtT = [[i,table[i][mediacol],table[i][titlecol].lower(),table[i][timecol]] for i in range(len(table))]
    imtT = sorted(imtT, key=lambda x: (x[1],x[2],-float(x[3])))
    # Keep only ones different from their successor, and the last one:
    OK = [imtT[i][1] != imtT[i+1][1] or imtT[i][2] != imtT[i+1][2] for i in range(len(imtT)-1)]+[True]
    #dup = [OK[0]]+[(not OK[i-1]) or (not OK[i]) for i in range(1,len(OK))]
    #idx = np.arange(len(OK))[dup]
    #savetsv("title_duplicates.tsv",[table[int(imtT[i][0])] for i in idx])
    idx = np.arange(len(OK))[OK]
    print(len(table)-len(idx),"title-duplicates deleted, leaving",len(idx),"rows")
    keep = [int(imtT[i][0]) for i in idx]
    keep.sort()
    return [table[k] for k in keep]

# Keep only the part between "//" and "/":
def strip_url(url):
    ur = url.replace("//www.","//")
    try: 
        i = ur.index("//")
    except:
        return ""
    u = ur[i+2:]
    try: 
        i = u.index("/")
    except:
        i = len(u)
    return u[:i]

regex = re.compile(
        r'^(?:http|ftp)s?://' # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
        r'localhost|' #localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
        r'(?::\d+)?' # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)

def url_valid(url): return re.match(regex,url) is not None

def rnd(x): return int(round(x))

def plural(i):
    if i>0: return "s"
    else: return ""

def secondsago(seconds):
    minutes = rnd(seconds/60)
    if minutes<2: return "{} second".format(rnd(seconds))+plural(seconds)+" ago"
    hours = rnd(minutes/60)
    if hours<2: return "{} minute".format(minutes)+plural(minutes)+" ago"
    return  "{} hour".format(hours)+plural(hours)+" ago"

def daysago(days):
    weeks = rnd(days/7)
    if weeks<2: return "{} day".format(rnd(days))+plural(days)+" ago"
    months = rnd(days/30)
    if months<2: return "{} week".format(weeks)+plural(weeks)+" ago"
    years = rnd(days/365)
    if years<2: return "{} month".format(months)+plural(months)+" ago"
    return "{} year".format(years)+plural(years)+" ago"

def timeago_text(totalsecsago): 
    if totalsecsago < 0:
        print("Article allegedly from",totalsecsago/3600,"hours in the future")
        totalsecsago = -totalsecsago + 3600
    if (totalsecsago > 24*3600): 
        when = daysago(np.floor(totalsecsago/(24*3600)))
    else:
        when = secondsago(np.floor(totalsecsago))
    return when

def timeago_text2(totalsecsago,oldtext):
    when = timeago_text(totalsecsago)
    if "?" in oldtext: when = when + "?"
    return when

# Example: gettime("2020-04-05T02:27:39","UTC")
def gettime(timestring,timezone):
    try:
        dt = timestring+"0" # To fix possibly buggy input data
        tu = datetime.datetime.fromisoformat(dt[:10]+"T"+dt[11:19]+".000000")
        t = pytz.timezone(timezone).localize(tu)
        gotTime = True
    except:
        t = "No time"
        gotTime = False
    return(t,gotTime)    

def seconds_since_y2k():
    server_timezone = "UTC" # AWS uses UTC
    now = pytz.timezone(server_timezone).localize(datetime.datetime.now())
    y2ktime = gettime("2000-01-01T00:00:00","UTC")[0]
    return (now-y2ktime).total_seconds()

# Sample usage: year_month_day("1967-05-05T00:30:00","2020-24-01T00:00:00")
def year_month_day(date_time_published,date_time_scraped):
    # For now, assume that all media outlets are on EDT (UTC-4). Fix later, based on media_name!
    server_timezone = "UTC" # # AWS uses UTC
    media_timezone  = "America/New_York"
    now = pytz.timezone(server_timezone).localize(datetime.datetime.now())
    (time,gotTime)     = gettime(date_time_published,media_timezone)
    if gotTime: return (time.year,time.month,time.day)
    (time,gotTime)     = gettime(date_time_scraped,media_timezone)
    if gotTime: return (time.year,time.month,time.day)
    else: return (0,0,0)

####################################################
##### Tools for topic processing
####################################################

# Find ancestors, i.e., node(s) above each topic in directed acyclic graph: 
def ancestordict(topics):
    topicnames = [t[0] for t in topics]
    return {t[0]:[anc for anc in t[8:8+int(t[6])] if anc in topicnames] for t in topics}
#def ancestordict(topics):
#    return {t[0]:list(t[8:8+int(t[6])]) for t in topics}

# Find parents, i.e., node(s) immediately above each topic in directed acyclic graph: 
def parentdict(topics):
    ancestors = ancestordict(topics)
    def prnts(anc): return [anc[i] for i in range(2,len(anc)-1) if not anc[i-1] in ancestors[anc[i]]] + [anc[-1]]
    return {t[0]:prnts(ancestors[t[0]]) for t in topics[1:]}

# Find children, i.e., node(s) immediately below each topic in directed acyclic graph: 
def childdict(topics,parents):
    def kids(topic): return [t[0] for t in topics[1:] if int(t[2])==1 and topic in parents[t[0]]]
    return {t[0]:kids(t[0]) for t in topics}

# Find descendants, i.e., taggable node blow each topic in directed acyclic graph: 
def descendantdict(topics):
    ancestors = ancestordict(topics)
    def dscndnts(topic): return [t[0] for t in topics[1:] if int(t[3])==1 and topic in ancestors[t[0]]]
    return {t[0]:dscndnts(t[0]) for t in topics}
    
# Keep only topics with show=1:    
def get_topicinfo(topics):
    def childnumbers(topicname): return [topicdict[child] for child in children[topicname]]
    tp = [t for t in topics[1:] if int(t[2])==1]
    topicnames = [t[0] for t in tp]  
    ancestors = ancestordict(tp)
    parents = parentdict(tp)
    children = childdict(tp,parents)
    descendants = descendantdict(tp)
    topicdict = {tp[i][0]:i for i in range(len(tp))}
    # (name,tagnum,UCname,LCname,nchildren,pop,children)
    topiclist = [[tp[i][0],topicdict.get(tp[i][0],-1),tp[i][4],tp[i][5],len(children[tp[i][0]]),float(tp[i][7]),tp[i][16]]+childnumbers(tp[i][0]) for i in range(len(tp))]
    cookiedict = {t[0]:t[16] for t in topics}
    #topiclist = [[tp[i][0],tagtopicdict.get(tp[i][0],-1),tp[i][4],tp[i][5],len(children[tp[i])),float(tp[i][7])]+[topicdict[n] for n in tp[i][8:8+int(tp[i][6])]] for i in range(len(tp))]
    savecsv("topiclist.csv",topiclist)
    return (topiclist,topicdict,ancestors,children,cookiedict)
    # tagtopics & ancestors used here in index.py
    # topiclist, showtopicdict, topicdict & children used in itn.py
    # Never used: parents, descendants, showtopics

# "Level" actually just means number of ancestors:
def validate_topics(topics):
    def not_int(x):
        try:
            i = int(x)
            return False
        except:
            return True
    def not_float(x):
        try:
            f = float(x)
            return False
        except:
            return True
    try: names = [t[0] for t in topics[1:]]
    except: 
        print("Blank line found")
        quit()
    error  = False
    i = 1
    for t in topics[1:]:
        if len(t) < 17: 
           print("Too few columns:",len(t))
           error = True
        else:
            (code,id,show,tag,Name,name,nparents,popularity,tag0,tag1,tag2,tag3,tag4,tag5,tag6,pt,cookieCode) = t[:17]
            error = False
            if code == "": 
               print("No code")
               error = True
            if not_int(show): 
               print("Invalid show flag",show)
               error = True
            if not_int(tag): 
               print("Invalid tag flag",tag)
               error = True
            if Name == "": 
               print("No Name")
               error = True
            if name == "": 
               print("No name")
               error = True
            if not_float(popularity): 
               print("Invalid popularity",popularity)
               error = True
            else:
                if int(show)>0 and popularity == 0:
                    print("Zero popularity",popularity)
                    error = True
            if not_int(nparents): 
               print("Invalid nparents",nparents)
               error = True
            else: nparents = int(nparents)
            try:
                parents = t[8:8+nparents]
                for p in parents:
                    if not p in names:
                        print("Invalid parent",p)
                        error = True
                for p in t[9+nparents:8+7]:
                    if p != "": 
                        print("Uncounted parent:",p)
                        error = True          
            except:
                print("Not all",nparents,"parents listed",t[8:])
                error = True
            if type(cookieCode) is str: 
                if len(cookieCode) != 2:
                   print("Cookie code not 2 characters:",cookieCode)
                   error = True
            else:  
               print("Invalid cookie code")
               error = True
        if error: 
           print("Line",i+1,":",t)
        i = i + 1
    if len(set(names))<len(names): # Find duplicates:
        print(len(names)-len(set(names)),"duplicates found:")
        seen = {}
        dupes = []
        for x in names:
            if x not in seen:
                seen[x] = 1
            else:
                if seen[x] == 1:
                    dupes.append(x)
                seen[x] += 1
        print(dupes)
    return


def validate_media(media):
    def not_int(x):
        try:
            i = int(x)
            return False
        except:
            return True
    try: names = [m[0] for m in media]
    except: 
        print("Blank line found")
        quit()
    error  = False
    i = 0
    for m in media:
        i += 1
        if len(m) < 10: 
           print("Too few columns:",len(m))
           error = True
        else:
            (mnemnonic,url,imgurl,name,score1,score2,score3,score4,score5,MCid) = m[:10]
            error = False
            if mnemnonic == "": 
               print("No mnemnonic")
               error = True
            if not url_valid(url):
               print("Invalid url:",url)
               error = True
            if imgurl!="" and not url_valid(imgurl): 
               print("Invalid imgurl:",imgurl)
               error = True
            if name == "": 
               print("No name")
               error = True
            if not_int(score1): 
               print("Invalid score 1",score1)
               error = True
            if not_int(score2): 
               print("Invalid score 2",score2)
               error = True
            if not_int(score3): 
               print("Invalid score 3",score3)
               error = True
            if not_int(score4): 
               print("Invalid score 4",score4)
               error = True
            if not_int(score5): 
               print("Invalid score 5",score5)
               error = True
            if not_int(MCid): 
               print("Invalid MCid",MCid)
               error = True
        if error: 
           print("Line",i+1,":",m)
    if len(set(names))<len(names): # Find duplicates:
        print(len(names)-len(set(names)),"duplicates found:")
        seen = {}
        dupes = []
        for x in names:
            if x not in seen:
                seen[x] = 1
            else:
                if seen[x] == 1:
                    dupes.append(x)
                seen[x] += 1
        print(dupes)
    return

####################################################
##### Tools for media processing
####################################################



