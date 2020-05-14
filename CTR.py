def split_term(data)
    docs = array(data['tweets'].apply(lambda x: x.split()))

    bigram = Phrases(docs, min_count=10)
    trigram = Phrases(bigram[docs])



    for idx in range(len(docs)):
        for token in bigram[docs[idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
                docs[idx].append(token)
        for token in trigram[docs[idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
                docs[idx].append(token)



    dictionary = Dictionary(docs)
    dictionary.filter_extremes(no_below=10, no_above=0.2)

    corpus = [dictionary.doc2bow(doc) for doc in docs]


    data['corpus'] = corpus
    data['docs'] = docs
    return data


def gradienD(train_ef, F, n, alpha, ld):
    p,q = Initef(train_ef, F)
    train = list2dic(train)
    for step in range(0,n):
        for u, i in train.items():
            samples = SelectNegativeSample(popularity,train_ef[u])
            for item , rui in samples.items():
                eui = rui - sum(p[u][f] * q[item][f] for f in range(0, F))
                for f in range(0,F):
                    p[u][f] += alpha * (q[item][f] * eui - ld * p[u][f])
                    q[item][f] += alpha * (p[u][f] * eui - ld * q[item][f])
        alpha *= 0.9
    return p,q


def LFM(train, p, q, N):
    Allrank = dict()
    rank =dict()
    for u in train.keys():
        rank.clear()
        for i in q.keys():
            if i not in train[u]:
                if i not in rank:
                    rank[i] = 0
                    for f in range(0,F):
                        rank[i] += p[u][f] * q[i][f]
        Allrank[u]=[]
        for item,pop in sorted(rank.items(),key = op.itemgetter(1), reverse = True)[0:N]:
            Allrank[u].append(item)
    return Allrank


def cossim(t1, t2)
    sum = 0
    sq1 = 0
    sq2 = 0
    for i in range(len(t1)):
        sum += t1[i] * t2[i]
    sq1 += pow(t1[i], 2)
    sq2 += pow(t2[i], 2)
    try:
        result = round(float(sum) / (math.sqrt(sq1) * math.sqrt(sq2)), 2)
    except ZeroDivisionError:
        result = 0.0

    return result