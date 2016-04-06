from __future__ import division
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocess import lemmatize, tokenize
from nltk.corpus import stopwords
import operator

stopwords = set(stopwords.words('english'))

# Returns unigrams, bigrams, trigrams dicts
# For example, `trigrams[1] = ['house', 'is', 'nice']`
def get_grams_indices(tokens):
    unigrams, bigrams, trigrams = [], [], []
    for i in range(len(tokens)):
        if i < len(tokens)-2:
            trigrams[i] = tokens[i:i+2]
        if i < len(tokens)-1:
            bigrams[i] = tokens[i:i+1]
        unigrams[i] = tokens[i]
    return unigrams, bigrams, trigrams

# Takes as input a list of doc tokens (nested list)
# For instance, with two docs: [[token_1, token_2], [token_3, token_4, token_5]]
def extract_features(doc_tokens):
    for tokens in doc_tokens:
        unigrams, bigrams, trigrams = get_grams_indices(tokens)
        # Need to extract features (length of word, entropy, etc.)
    return features # Same length as doc_tokens (number of docs)

def valid_ngram(ngram):
    grams = ngram.split()
    # remove ngram that start and end with stopwords
    if grams[0] in stopwords and grams[-1] in stopwords:
        return ""
    # other heuristics for filtering goes here...

    return grams

# input: list of docs as strings: ['doc 1 string', 'doc 2 string']
# output: tfidf matrix, each row a doc, each col a phrase, each cell a tfidf score
#         list of vocabulary in the same order as features
#         record of first occurrence of each valid ngram in each doc
def get_tfidf_matrix(docs):
    first_occurrence_all = []

    vectorizer = TfidfVectorizer(preprocessor=lemmatize, ngram_range=(1, 3), tokenizer=tokenize)
    analyze = vectorizer.build_analyzer()
    # construct our own vocab applying some heuristics
    vocab = set()
    for doc in docs:
        first_occurrence = {}
        tokenized_doc = analyze(doc)
        total = len(tokenized_doc)
        for i, ngram in enumerate(tokenized_doc):
            if ngram not in first_occurence:
                first_occurrence[ngram] = i / total
            if valid_ngram(ngram):
                vocab.add(ngram)
        first_occurrence_all.append(first_occurrence)

    vocab = list(vocab)

    vectorizer.vocabulary=vocab
    X = vectorizer.fit_transform(docs)

    # get list of phrases in the order of the feature vector
    vocab_list = [phrase for phrase, idx in sorted(vectorizer.vocabulary_.items(), key=operator.itermgetter(1))]
    assert(len(vocab_list) == X.shape[1])

    return X, vocab_list, first_occurrence_all

def get_first_occurrence(phrase, docid, tokenized_docs):
    score = 0 # should be normalized to be in (0, 1)

    return score

# input: tfidf_matrix, list of all phrases in vocab, set of all true keywords for each doc
# output: feature matrix: [[feature vector1], [feature vector2], ...], labels:[0, 1, ...]
def get_feature_vectors(tfidf_matrix, phrase_list, true_keys, first_occurrence):
    num_features = 2 # might add more features
    #features = np.empty([0, num_features], dtype='float32')
    features = []
    labels = []

    # tfidf matrix
    doc_vecs = tfidf_matrix.toarray().tolist()

    for docid, vec in enumerate(doc_vecs):
        # traverse the doc vector
        for i, tfidf in enumerate(vec):
            if tfidf != 0:
                feature = []
                feature.append(tfidf)
                feature.append(first_occurrence[docid][phrase_list[i]])
                # add more features...

                features.append(feature)

                # append label
                if phrase in true_keys[docid]:
                    labels.append(1)
                else:
                    labels.append(0)

    return np.array(features), labels
