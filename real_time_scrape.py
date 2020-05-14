
import json
import csv
import tweepy
import re
import pandas as pd
import time

from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
from tweepy.models import Status

import sys

import warnings
warnings.filterwarnings('ignore')
consumer_key = "**********************"
consumer_secret = "**********************"
access_key = "**********************"
access_secret = "**********************"




auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)




class listener(StreamListener):
    def on_data(self, data):
        with open('full_03171020.txt', 'a+') as file:
            w = csv.writer(file)
            w.writerow([data])

        return (True)

    def on_error(self, status):
        print(status)


auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)

twitterStream = Stream(auth, listener())
twitterStream.filter(locations=[-74, 40, -73, 41],
                     languages=['en'])


def create_csv(fname):
    path = fname + ".csv"
    with open(path,'w') as f:
        csv_write = csv.writer(f)
        csv_write.writerow([ 'username', 'timestamp',
           'tweet_text', 'all_hashtags',
           'user_mentions', 'location',
           'retweet_count', 'favorite_count',
           'lang'])


def get_tweets_to_csv(username, fname):
    # Authorization to consumer key and consumer secret
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

    # Access to user's access key and access secret
    auth.set_access_token(access_key, access_secret)

    # Calling api
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    # 200 tweets to be extracted
    numberOfUsersMined = 0
    with open('%s.csv' % (fname), 'a+', encoding="utf8", errors='ignore') as file:
        w = csv.writer(file)

        # for each tweet matching our hashtags, write relevant info to the spreadsheet
        for tweet in tweepy.Cursor(api.user_timeline,
                                   screen_name=username, include_rts=True,
                                   tweet_mode='extended').items(30):
            w.writerow([tweet.user.screen_name,
                        tweet.created_at,
                        tweet.full_text.replace('\n', ' '),
                        " ".join([e['text'] for e in tweet._json['entities']['hashtags']]),
                        " ".join([m['screen_name'] for m in tweet._json['entities']['user_mentions']]),
                        tweet.retweet_count,
                        tweet.favorite_count,
                        tweet.lang])
            numberOfUsersMined += 1
            if numberOfUsersMined % 90 == 0:
                time.sleep(15)
