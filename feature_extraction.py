from __future__ import division
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocess import lemmatize, tokenize
from nltk.corpus import stopwords
import operator

stopwords = set(stopwords.words('english'))
features = ['tfidf', 'docs', 'max_length']

# Returns unigrams, bigrams, trigrams dicts
# For example, `trigrams[1] = ['house', 'is', 'nice']`
# def get_grams_indices(tokens):
#     unigrams, bigrams, trigrams = [], [], []
#     for i in range(len(tokens)):
#         if i < len(tokens)-2:
#             trigrams[i] = tokens[i:i+2]
#         if i < len(tokens)-1:
#             bigrams[i] = tokens[i:i+1]
#         unigrams[i] = tokens[i]
#     return unigrams, bigrams, trigrams
#

# Takes as input a list of doc tokens (nested list)
# For instance, with two docs: [[token_1, token_2], [token_3, token_4, token_5]]
def extract_features(docs, keys):
    tfidf_matrix, phrase_list, first_occurrence_all = get_tfidf_matrix(docs)
    X, y = get_feature_matrix(tfidf_matrix, phrase_list, keys, first_occurrence_all)
    return X, y

# remove ngrams that start and end with stopwords
def valid_ngram(ngram):
    grams = ngram.split()
    if grams[0] in stopwords or grams[-1] in stopwords:
        return False
    # other heuristics for filtering go here...
    return True

# input: list of docs as strings: ['doc 1 string', 'doc 2 string']
# output: tfidf matrix, each row a doc, each col a phrase, each cell a tfidf score
#         list of vocabulary in the same order as features
#         record of first occurrence of each valid ngram in each doc
def get_tfidf_matrix(docs):
    first_occurrence_all = []
    vectorizer = TfidfVectorizer(decode_error='ignore', preprocessor=lemmatize, ngram_range=(1, 3), tokenizer=tokenize)
    analyze = vectorizer.build_analyzer()
    #preprocessor = vectorizer.build_preprocessor()
    #tokenizer = vectorizer.build_tokenizer()
    # construct our own vocab applying some heuristics
    vocab = []
    print "learning vocabulary"
    for doc in docs:
        first_occurrence = {}
        tokenized_doc = analyze(doc)
        total = len(tokenized_doc)
        for i, ngram in enumerate(tokenized_doc):
            if ngram not in first_occurrence:
                first_occurrence[ngram] = float(i) / total
            if valid_ngram(ngram):
                vocab.append(ngram)
        first_occurrence_all.append(first_occurrence)
    print "size of vocabulary: ", len(vocab)
    print "transforming tfidf matrix"
    vectorizer.vocabulary = list(vocab)
    X = vectorizer.fit_transform(docs)
    # get list of phrases in the order of the feature vector
    vocab_list = [phrase for phrase, idx in sorted(vectorizer.vocabulary_.items(), key=operator.itemgetter(1))]
    assert(len(vocab_list) == X.shape[1])
    return X, vocab_list, first_occurrence_all
    #return preprocessor, tokenizer, analyze


# Input: parameters to determine features
# Ouput: feature vector for a single keyphrase of size len(features)
def get_feature_vector(tfidf, first_occurence, doc_id, phrase):
    feature_vec = np.array()
    for f in features:
        if f == 'tfidf':
            feature_vec = np.append(feature_vec, tfidf)
        elif f == 'first_occurence':
            feature_vec = np.append(feature_vec, first_occurence[doc_id][phrase])
    return feature_vec

# input: tfidf_matrix, list of all phrases in vocab, set of all true keywords for each doc
# output: feature matrix (np.array): [[feature vector1], [feature vector2], ...], labels:[0, 1, ...]
def get_feature_matrix(tfidf_matrix, phrase_list, true_keys, first_occurrence):
    X = np.empty((0, len(features)))
    y = np.empty(0)
    doc_tfidf_vecs = tfidf_matrix.toarray().tolist() # tfidf matrix

    for doc_id, tfidf_vec in enumerate(doc_tfidf_vecs):
        # traverse the doc vector
        for i, tfidf in enumerate(tfidf_vec):
            if tfidf != 0: # Why is this case here?
                feature_vec = get_feature_vector(tfidf, first_occurence, doc_id, phrase_list[i])
                X = np.append(X, feature_vec, axis=0)
                label = lambda: 1 if phrase in true_keys[doc_id] else 0
                y = np.append(y, label())
    return X, y

def get_vec_differences(X_vec, y_vec):
    X = np.empty((0, np.size(X_vec, axis=1)))
    y = np.empty(0)
    for i in range(len(X_vec)):
        for j in range(i, len(X_vec)):
            if y_vec[i] == y_vec[j]:
                continue
            elif y_vec[i] > y_vec[j]:
                X = np.append(X, X_vec[i] - X_vec[j], axis=0)
                y = np.append(y, 1)
            elif y_vec[i] < y_vec[j]:
                X = np.append(X, X_vec[i] - X_vec[j], axis=0)
                y = np.append(y, 0)
    return X, y
# def construct_feature_vectors(train_docs):
#     for i in range(len(train_docs)):
#         # Dict of {index of gram : list of words in gram}
#         train_grams = get_grams(train_docs[i])
#         for g in train_grams:
#             if g in train_keys: # train_keys needs to be normalized I think
#                 is_keyword = True
#             else:
#                 is_keyword = False
#             train_data.append(extract_features(g, train_grams, is_keyword))
#     for i in range(len(test_docs)):
#         # Dict of {index of gram : list of words in gram}
#         test_grams = get_grams(test_docs[i])
#         for g in test_grams:
#             if g in test_keys: # test_keys needs to be normalized I think
#                 is_keyword = True
#             else:
#                 is_keyword = False
#             test_data.append(extract_features(g, test_grams, is_keyword))
#     # train_vec should probably be the first element of test_data, and the label
#     # can be the second element of test_data. both are output by extract_features
#     # above in the loop
