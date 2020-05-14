import operator
import collections

def corpus(data):
    word_list = []
    for tweet in data.clean_tweet:
        for word in tweet.split(' '):
            word_list.append(word)
    tweet_corpus = ' '.join(word_list)


    hash_list = []
    for hashtags in data['all_hashtags']:
        if hashtags != ():
            for hashtag in hashtags:
                hash_list.append(hashtag)

    hash_corpus = ' '.join(hash_list)

    all_corpus = str.lower(tweet_corpus + " " + hash_corpus)
    WF = collections.Counter(all_corpus.split())
    HF = collections.Counter(hash_corpus.split())
    return WF, HF

#%%

def hmaps(data):
    hmaps = {}
    for i in range(data_hash.shape[0]):
        text_process(data_hash.all_hashtags.iloc[i])
        for j in data_hash.all_hashtags.iloc[i]:
            terms = [text for text in data_hash.clean_tweet.iloc[i].split(' ')]
            if j in hmaps:
                for t in terms:
                    hmaps[j].append(t)
            else:
                hmaps[j] = [terms[0]]
                for t in terms[1:]:
                    hmaps[j].append(t)
    return hmaps

#%%

def get_key(val,my_dict):
    keys = []
    for key, value in my_dict.items():
        if val in value:
            keys.append(key)
    return keys

#%%

def hashtag_recommend(tweet,WF,hmaps,N):
    tweet = clean_tweets(p.clean(tweet))
    tweet = set(text_process(tweet))
    Sj = {}
    for word in tweet:
        h = get_key(word,hmaps)
        thfm_sum = sum([WF[str.lower(j)] for j in h])
        q = [hmaps[j] for j in h]
        hfm_sum = sum([WF[str.lower(x)] for j in q for x in j])
        for j in h:
            hf = WF[str.lower(j)]/thfm_sum
            ihu = np.log(len(word_list)/hfm_sum)
            #print(hf, ihu)
            if j in Sj.keys():
                Sj[j] +=(hf*ihu)
            else:
                Sj[j] = (hf*ihu)
    Sj = sorted(Sj.items(), key=operator.itemgetter(1),reverse=True)
    return Sj[:N]