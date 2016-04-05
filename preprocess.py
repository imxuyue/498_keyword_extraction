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

def tokenize(doc):
    # Tokenize something
    return doc


# input: a list of strings representing a doc
# output: a list of lemmatized strings
def lemmatize(doc):
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(token) for token in doc.split()])

# input:  a list l of string
# output: a list containing the stemmed string in l
def stem(doc):
    stemmer = PorterStemmer()
    # r_tokens is a list of tokens in one review
    return ' '.join([stemmer.stem(token) for token in doc.split()])

# input:  a list l of string
# output: a list of string where the stopwords are removed
def remove_stopwords(doc):
    '''
    for token in l:
        if token not in stopwords:
            result.append(token)
    '''
    return ' '.join([token for token in doc.split() if token not in stopwords])


