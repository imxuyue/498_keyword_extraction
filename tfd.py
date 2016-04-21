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


class TFD(object):
    def __init__(self, data_dir, rank = 36):
        self.data_dir = data_dir
        self.stopwords = stopwords.words('english')
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.rank = rank

    def pre_process(self, text):
        text = re.sub(r'[^a-z-]+', ' ', text)
        text = re.split(r'[^a-z-]+', text)
        text = [word for word in text if word not in self.stopwords]
        text = [self.stemmer.stem(word) for word in text]
        text = [self.lemmatizer.lemmatize(word) for word in text]
        text = [word for word in text if len(word) > 1]
        return text

    def get_doc(self, f):
        self.doc = f.read().lower()
        self.doc = self.pre_process(self.doc)

    def get_key(self, f):
        self.key = []
        for line in f.readlines():
            line = self.pre_process(line.lower())
            line = ' '.join(line)
            self.key.append(line)

    def append_ngrams(self, tokens):
        temp = []
        bigram = [' '.join(tokens[i:i+2]) for i in xrange(len(tokens) - 1)]
        trigram = [' '.join(tokens[i:i+3]) for i in xrange(len(tokens) - 2)]
        fourgram = [' '.join(tokens[i:i+4]) for i in xrange(len(tokens) - 3)]
        temp = tokens + bigram + trigram + fourgram
        return temp

    def partition_doc(self, n = 5):
        self.d = dict()
        bin = len(self.doc) / n  #number of terms in each part
        for i in xrange(n):
            self.d[i] = self.doc[i*bin : (i+1)*bin]
        self.d[n-1] = self.doc[(i+1)*bin : ]
        for i in xrange(n):
            self.d[i] = self.append_ngrams(self.d[i])
        return self.d

    def get_term_frequency(self):
        self.tf = dict()
        tokens = self.append_ngrams(self.doc)
        for term in tokens:
            self.tf[term] = self.tf.get(term, 0) + 1
        return self.tf

    def get_term_distribution(self):
        self.td = dict()
        self.partition_doc()
        for term in set(self.append_ngrams(self.doc)):
            for key in self.d:
                if term in self.d[key]:
                    self.td[term] = self.td.get(term, 0) + 1
        return self.td

    def get_term_score(self):
        if self.method == 'tf':
            self.score = self.get_term_frequency()
        if self.method == 'td':
            self.score = self.get_term_distribution()
        if self.method == 'tfd':
            self.get_term_distribution()
            self.get_term_frequency()
            self.score = {key:self.tf[key] + self.td[key] for key in self.td}

    def get_ans(self, ans_file):
        with open(self.data_dir + ans_file, 'r') as f:
            self.ans = []
            for line in f.readlines():
                line = self.pre_process(line.lower())
                line = ' '.join(line)
                self.ans.append(line)
            self.ans = [word for word in self.ans if word in self.append_ngrams(self.doc)]

    def fit(self, filename, ans_file):
        self.get_keywords(filename, self.method)
        self.get_ans(ans_file)

    def get_keywords(self, filename, method):
        self.method = method
        with open(self.data_dir + filename, 'r') as f:
            self.get_doc(f)
            self.get_term_score()
            doclen = len(set(self.append_ngrams(self.doc)))
            self.kw = [items[0] for items in sorted(self.score.items(), key = lambda item: item[1])[-doclen/self.rank:]]
            return self.kw

    def get_accuracy_recall(self):
        hit = len([word for word in self.kw if word in self.ans])
        accuracy = hit*1.0 / len(self.kw)
        recall = hit*1.0 / len(self.ans)
        return accuracy, recall

    def get_accuracy_recall2(self, method):
        self.method = method
        accuracy = recall = 0.0
        nfiles = 0.0
        for filename in os.listdir(self.data_dir):
            if filename.endswith('txt'):
                nfiles += 1
                ans_file = filename[:-3] + 'key'
                self.fit(filename, ans_file)
                a, r = self.get_accuracy_recall()
                accuracy += a
                recall += r 
        return accuracy/nfiles, recall/nfiles

