from __future__ import division
import nltk
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string

stopwords = set(stopwords.words('english'))

def tokenize(doc):
    res = doc
    # punct_regex = re.compile('[%s]' % re.escape(string.punctuation))
    # l = []
    # for token in res.split():
    #     token = re.sub(punct_regex, '', token)
    #     if token:
    #         l.append(token)
    l = res.split()

    return l

# input: a list of strings representing a doc
# output: a list of lemmatized strings
def lemmatize(doc):
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(token) for token in doc.split()])

def preprocess(res):
    #res = res.decode('utf-8', 'ignore')
    # matches all punctuation except for '-' and single quote
    punct_regex = re.compile('[%s]' % re.escape('!"#$%&\()*+,./:;<=>?@[\\]^_`{|}~'))
    # lemmatize
    res = lemmatize(res)
    # get rid of all punctuation
    res = re.sub(punct_regex, '', res)
    # remove any number
    res = re.sub(r"(?u)\b\d+\b", "", res)
    # remove some punctuations followed by space
    #res = re.sub(r"(\w+)([,.?!:;])(\s|$)", r"\1\3", res)

    # get rid of double quotes
    #res = re.sub(r"\"", "", res)

    # get rid of parentheses
    #res = re.sub(r'[\[\]{}()|*%$#@&+-,.]+', "", res)
    res = res.lower()
    res = ' '.join(res.split())
    return res


# input:  a list l of string
# output: a list containing the stemmed string in l
def stem(doc):
    stemmer = PorterStemmer()
    # r_tokens is a list of tokens in one review
    return ' '.join([stemmer.stem(token) for token in doc.split()])

# input:  a list l of string
# output: a list of string where the stopwords are removed
def remove_stopwords(doc):
    # for token in l:
    #     if token not in stopwords:
    #         result.append(token)
    return ' '.join([token for token in doc.split() if token not in stopwords])
