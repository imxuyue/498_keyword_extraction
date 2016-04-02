'''
preprocess.py
-------------
Yue Xu
Mar 20, 2016
'''

from __future__ import division
import nltk
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

stopwords = stopwords.words('english')

def tokenize(l):

    return 0


# input: a list of strings representing a doc
# output: a list of lemmatized strings
def lemmatize(l):
    lemmatizer = WordNetLemmatizer()
    result = []
    for doc in l:
        tokens = [lemmatizer.lemmatize(token) for token in doc.split()]
        result.append(' '.join(tokens))
    return result


# input:  a list l of string
# output: a list containing the stemmed string in l
def stem(l):
    result = []
    stemmer = PorterStemmer()

    for doc in l:
        # r_tokens is a list of tokens in one review
        d_singles = [stemmer.stem(token) for token in doc.split()]
        result.append(' '.join(d_singles))

    return result

# input:  a list l of string
# output: a list of string where the stopwords are removed
def removeStopwords(l):

    result = []

    '''
    for token in l:
        if token not in stopwords:
            result.append(token)
    '''
    for doc in l:
        d_no_sw = [token for token in doc.split() if token not in stopwords]
        result.append(' '.join(d_no_sw))

    return result


