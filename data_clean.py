import pandas as pd
import numpy as np
import json
import csv


def full_tweet(tweet):
    r = tweet.get('extended_tweet',None)
    if r != None:
        r = r['full_text']
    else:
        r = tweet['text']
    return r

def quote_tweet(tweet):
    r = tweet.get('quoted_status',None)
    if r != None:
        rr = r.get('extended_tweet',None)
        user = r['user']['screen_name']
        if rr != None:
            rr = rr['full_text']
        else:
            rr = r['text']
    else:
        rr = r
        user = r
    return rr, user

import datetime
import pytz

def convert_datetime_timezone(dt, tz1 = 'UTC', tz2 = 'EST'):
    tz1 = pytz.timezone(tz1)
    tz2 = pytz.timezone(tz2)

    dt = datetime.datetime.strptime(dt,"%a %b %d %H:%M:%S %z %Y")
#    dt = tz1.localize(dt)
    dt = dt.astimezone(tz2)
    dt = dt.strftime("%a %b %d %H:%M:%S %Y")

    return dt


from datetime import datetime as dt

def rtime_clean(raw_statue):
    data = pd.read_csv(raw_statue,header = None)
    my_json = data[0].map(lambda x:json.loads(x))
    df = pd.DataFrame(columns = ['screen_name', 'timestamp','tweet_id',
                            'tweet_text','quoted_status','quoted_user','all_hashtags', 'all_hashtags2',
                            'user_mentions','user_mentions2','retweet_count', 'favorite_count',
                            'lang','place_fullname','place_coordinates_average',
                            'user_name','user_location','user_description','user_followers_count',
                            'user_friends_count','user_statuses_count'])
    df['screen_name'] = my_json.map(lambda x: x['user']['screen_name'])
    df['timestamp'] = pd.to_datetime(my_json.map(lambda x: convert_datetime_timezone(x['created_at'])))
    df['hour'] = [dt.strftime(x,'%H:%M') for x in df['timestamp']]
    df['day'] = [dt.strftime(x,'%m-%d') for x in df['timestamp']]
    df['tweet_id'] = my_json.map(lambda x: x['id'])
    df['tweet_text'] = my_json.map(lambda x: full_tweet(x))
    df['quoted_status'] = my_json.map(lambda x: quote_tweet(x)[0])
    df['quoted_user'] = my_json.map(lambda x: quote_tweet(x)[1])
    df['all_hashtags'] = my_json.map(lambda x: tuple([i['text'] for i in x['entities']['hashtags']]))
    df['all_hashtags2'] = my_json.map(lambda x: ', '.join([i['text'] for i in x['entities']['hashtags']]))
    df['user_mentions'] = my_json.map(lambda x: tuple([i['screen_name'] for i in x['entities']['user_mentions']]))
    df['user_mentions2'] = my_json.map(lambda x: ', '.join([i['screen_name'] for i in x['entities']['user_mentions']]))
    df['retweet_count'] = my_json.map(lambda x: x['retweet_count'])
    df['favorite_count'] = my_json.map(lambda x: x['favorite_count'])
    df['lang'] = my_json.map(lambda x: x['lang'])
    df['place_fullname'] = my_json.map(lambda x: x['place']['full_name'])
    df['place_coordinates_average'] = my_json.map(lambda x: np.mean(x['place']['bounding_box']['coordinates'][0],axis = 0))
    df['lon'] = df['place_coordinates_average'].map(lambda x:x[1])
    df['lat'] = df['place_coordinates_average'].map(lambda x:x[0])
    df['user_name'] = my_json.map(lambda x: x['user']['name'])
    df['user_location'] = my_json.map(lambda x: x['user']['location'])
    df['user_description'] = my_json.map(lambda x: x['user']['description'])
    df['user_followers_count'] = my_json.map(lambda x: x['user']['followers_count'])
    df['user_friends_count'] = my_json.map(lambda x: x['user']['friends_count'])
    df['user_statuses_count'] = my_json.map(lambda x: x['user']['statuses_count'])
    return df


emoticons_happy = set([
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
    '<3'
])

emoticons_sad = set([
    ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '>:\\', ';('
])

emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)

emoticons = emoticons_happy.union(emoticons_sad)

# %%


from textblob import TextBlob
import string
from tweet_preprocessor import preprocessor as p
import emoji

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk

from nltk.corpus import stopwords

nltk.download('stopwords')

from nltk.tokenize import word_tokenize

nltk.download('punkt')

from nltk.tokenize import TweetTokenizer

tweet_tokenizer = TweetTokenizer()


def text_process(mess):
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]
    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


def clean_tweets(tweet):
    tweet = emoji.demojize(tweet)
    # print(tweet)
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(tweet)
    punc = string.punctuation + '—’...’…・・・•“”""'
    # after tweepy preprocessing the colon symbol left remain after      #removing mentions

    tweet = re.sub(r':', '', tweet)
    # print(tweet)
    # tweet = re.sub(r'‚Ä¶', '', tweet)
    # replace consecutive non-ASCII characters with a space
    tweet = re.sub(r'[^\x00-\x7F]+', ' ', tweet)
    # print(tweet)
    # remove emojis from tweet
    tweet = emoji_pattern.sub(r'', tweet)
    # print(tweet)
    # filter using NLTK library append it to a string
    filtered_tweet = []
    # looping through conditions
    for w in word_tokens:
        # check tokens against stop words , emoticons and punctuations
        if w not in stop_words and w not in emoticons and w not in punc:
            filtered_tweet.append(str.lower(w))
    return ' '.join(filtered_tweet)
    # print(word_tokens)
    # print(filtered_sentence)return tweet

# %%
