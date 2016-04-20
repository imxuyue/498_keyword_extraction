'''
term frequency and term term distribution based keyword extraction
'''

import re
import os
import nltk
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import defaultdict

stopwords = stopwords.words('english')
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
data_dir = 'data/train2/'


def pre_process(text):
    text = re.sub(r'[^a-z-]+', ' ', text)
    text = re.split(r'[^a-z-]+', text)
    text = [word for word in text if word not in stopwords]
    text = [stemmer.stem(word) for word in text]
    text = [lemmatizer.lemmatize(word) for word in text]
    text = [word for word in text if len(word) > 1]
    return text

def get_doc(f):
    doc = f.read().lower()
    doc = pre_process(doc)
    return doc

def get_key(f):
    key = []
    for line in f.readlines():
        line = pre_process(line.lower())
        line = ' '.join(line)
        key.append(line)
    return key

def get_doc_key(data_dir, filename):
    with open(data_dir + filename, 'r') as f:
        doc = get_doc(f)
    with open(data_dir + filename[:-3] + 'key', 'r') as f:
        key = get_key(f)
    return doc, key

def append_ngrams(tokens):
    bigram = [' '.join(tokens[i:i+2]) for i in xrange(len(tokens) - 1)]
    trigram = [' '.join(tokens[i:i+3]) for i in xrange(len(tokens) - 2)]
    fourgram = [' '.join(tokens[i:i+4]) for i in xrange(len(tokens) - 3)]
    tokens.extend(bigram + trigram + fourgram)
    return tokens


def partition_doc(doc, n = 3):
    d = dict()
    bin = len(doc) / n  #number of terms in each part
    for i in xrange(n):
        d[i] = doc[i*bin : (i+1)*bin]
    d[n-1] = doc[(i+1)*bin : ]
    for i in xrange(n):
        d[i] = append_ngrams(d[i])
    return d

def get_term_frequency(doc):
    tf = dict()
    for term in doc:
        tf[term] = tf.get(term, 0) + 1
    return tf

def get_term_distribution(d, doc):
    td = dict()
    for term in set(doc):
        for key in d:
            if term in d[key]:
                td[term] = td.get(term, 0) + 1
    return td

def get_accuracy_recall(d, key, doclen, rank = 9):
    kw = [items[0] for items in sorted(d.items(), key = lambda item: item[1])[-doclen/rank:]]
    hit = len([word for word in kw if word in key])
    accuracy = hit * 1.0 / len(kw)
    recall = hit * 1.0 / len(key)
    return accuracy, recall

if __name__ == "__main__":
    accuracy = defaultdict(list)
    recall = defaultdict(list)
    nhit = nkey = 0.0
    nfiles = 0

    for rank in [1,3,9,18, 36, 72]:
        for filename in os.listdir(data_dir):
            if filename.endswith('txt'):
                nfiles += 1
                doc, key = get_doc_key(data_dir, filename)
                pdoc = partition_doc(doc, 5)
                doc = append_ngrams(doc)
                doclen = len(set(doc))
                key = [term for term in key if term in doc]

                tf = get_term_frequency(doc)
                td = get_term_distribution(pdoc, doc)
                tfd = {key:tf[key] + td[key] for key in td}

                methods = {'tf':tf, 'td':td, 'tfd':tfd}
                for method, val in methods.items():
                    a, r = get_accuracy_recall(val, key, doclen, rank)
                    accuracy[method].append(a)
                    recall[method].append(r)
        for key in accuracy:
            print sum(accuracy[key]) / nfiles, sum(recall[key]) / nfiles
        print

            







    

# acc = count = 0.0
# thit = tkey = 0.0

# for filename in os.listdir(data_dir):
#     if filename.endswith('txt'):
#         count += 1
#         doc, key = process(data_dir, filename)
#         key = [word for word in key if word in doc]
#         score = dict()
#         n = len(doc)
#         for term in doc:
#             score[term] = score.get(term, 0) + 1
#             if doc.index(term) <= n/3 or doc[::-1].index(term) <= n/3:
#                 score[term] = score.get(term, 0) + 1
#             # elif doc.index(term) <= n/3:
#             #     score[term] = score.get(term, 0) + 0
#             # elif doc[::-1].index(term) <= n/3:
#             #     score[term] = score.get(term, 0) + 0
#         kw = [items[0] for items in sorted(score.items(), key = lambda item: item[1], reverse=True)][:len(doc)/9]
#         hit = len([word for word in kw if word in key])
#         acc += hit*1.0 / len(kw)
#         thit += hit
#         tkey += len(key)
# print acc / count, thit/tkey  



