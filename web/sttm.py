
#pip3 install git+https://github.com/rwalk/gsdmm.git
# python3 ./congress/sttm.py

import pickle
import matplotlib as plt
import pandas as pd
import numpy as np
import ast
import texthero as hero
from texthero import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from biterm.utility import vec_to_biterms, topic_summuary # helper functions
# STTM lib from Github
from gsdmm import MovieGroupProcess
import re
import pyLDAvis


##########FUNCTIONS#######

def top_words(cluster_word_distribution, top_cluster, values):
    for cluster in top_cluster:
        sort_dicts =sorted(mgp.cluster_word_distribution[cluster].items(), key=lambda k: k[1], reverse=True)[:values]
        print('Cluster %s : %s'%(cluster,sort_dicts))
        print(' — — — — — — — — — ')

####LOAD AND CLEAN TEXT######

# exec(open("/Users/essbie/Desktop/CodingProjects/lasocial/sttmclean.py").read())
exec(open("/Users/essbie/Desktop/CodingProjects/lasocial/congress/csttmclean.py").read())

#kept coming up with weird index issues trying to fix it with this
df = df.reset_index(drop=True)

docs = df["toke"]

import random
random.seed(1000)
# Train STTM model
K = 27
mgp = MovieGroupProcess(K, alpha=0, beta=0.01, n_iters=50)

vocab = set(x for doc in docs for x in doc)
n_terms = len(vocab)
y = mgp.fit(docs, n_terms)

# #########################VISUALIZATION###################
# for whatever reason, this seems to work better before you do any further processing
import pandas as pd
import pyLDAvis
import math

doc_lengths = [len(doc) for doc in docs]
exec(open("/Users/essbie/Desktop/CodingProjects/lasocial/congress/sttmviz.py").read())


###Other processing

doc_count = np.array(mgp.cluster_doc_count)
print('Number of documents per topic :', doc_count)
print('*'*20)

# Topics sorted by the number of document they are allocated to
top_index = doc_count.argsort()[-10:][::-1]
print('Most important clusters (by number of docs inside):', top_index)
print('*'*20)
# Show the top 5 words in term frequency for each cluster
top_words(mgp.cluster_word_distribution, top_index, 10)

# te = []
# for i in df["toke"]:
#     a = max(mgp.score(i))
#     if a < 1:
#         te.append(99)
#     else:
#         k = mgp.choose_best_label(i)
#         te.append(k[0])
df["topic"] = y
df["topic"] = df["topic"]+1
df['enga'] = df['replyCount']+df['retweetCount']+df['likeCount']+df['quoteCount']

########DEFINE FILTER FOR TOPIC NUMBER#################
def ts(df,num):
    df["cvs"]=df['topic']==num
    justCVS = df[df['cvs']==True]
    justCVS.sort_values(['enga'],ascending=False)
    a = str(len(justCVS))
    b = str((len(justCVS)/len(df))*100)
    c = justCVS.sample(10)
    print("Sample of 10 tweets from this topic:")
    print(c.content.values + " [" + c.url.values + "]")
    print("Total Tweets In Topic: " + a)
    print("Topic is: " + b + " % of all Tweets")

#putting this here to remind you
print('Most important clusters (by number of docs inside):', top_index)

topic=[]
for i in df["topic"]:
    topic.append(str(i).zfill(2))

df['topic'] = topic

f = df.groupby(['topic'], as_index=False).agg(engaSum = ('enga','sum'))
# f = df.groupby(['topic'], as_index=False).size()
# .sort_values(ascending=False)
df = pd.merge(df,f,how='left',on='topic')
df = df.sort_values(by=['enga'],ascending=False)

# df.to_csv("./TwitClips/web/topics.csv")
df.to_csv("./congress/web/topics.csv")

#to scan for specific terms
def keyScan(finalTweets,term):
    finalTweets = df
    cvs2=[]
    for j in finalTweets['content']:
        if all(x in j for x in [term]) is True:
            cvs2.append("Y")
        else:
            cvs2.append("N")
    finalTweets["F"]=cvs2
    j2 = finalTweets[finalTweets["F"]=="Y"]
    engSum = sum(j2.enga)
    percent = engSum/sum(df.enga)
    print("Total Engagements: ")
    print(engSum)
    print("Percent of Total Tweets: ")
    print(percent*100)
    global ksRes
    ksRes = j2


# import snscrape.modules
# import datetime
# from datetime import timedelta
# from datetime import date
# refreshed = datetime.datetime.now().now()
#
# #just a lazy holdover from another script
# yesterday = refreshed - timedelta(1)
#
# print("Getting data from Twitter...")
# tOut = []
# useDate=yesterday.strftime("%Y-%m-%d")
# # data = "'url:https://medicareadvocacy.org/report-snf-financial-support-during-covid/ since:"+useDate+"'"
# data = "'#carecantwait since:"+useDate+"'"
# # data = "'caregiving_economy since:"+useDate+"'"
# tweets = snscrape.modules.twitter.TwitterSearchScraper(data)
# for j in tweets.get_items():
#     tOut.append(j)
#
# fd = pd.DataFrame(tOut)
