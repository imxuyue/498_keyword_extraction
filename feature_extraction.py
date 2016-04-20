from __future__ import division
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocess import preprocess, lemmatize, tokenize
from nltk.corpus import stopwords
import operator
import cPickle as pickle
from math import log10

stopwords = set(stopwords.words('english'))
features = ['tfidf', 'first_occurrence', 'entropy', 'length', 'num_tokens']

# Takes as input a list of doc tokens (nested list)
# For instance, with two docs: [[token_1, token_2], [token_3, token_4, token_5]]
def extract_features(docs, keys):
    tfidf_matrix, phrase_list, first_occurrence_all, entropy_all = get_tfidf_matrix(docs)
    #print "calculating entropy for phrases"
    #entropy_all = [get_phrase_entropy(doc, phrase_list) for doc in docs]

    X, y = get_feature_matrix(tfidf_matrix, phrase_list, keys, first_occurrence_all, entropy_all)
    return X, y

# extract candidates from a single doc
# phrase_list and df_vec are from training set
def extract_candidates_doc(doc, phrase_list, df_vec, training_size = 450):

    #vocab = set(phrase_list)
    df_dic = {}
    for i, phrase in enumerate(phrase_list):
        df_dic[phrase] = df_vec[i]
    noun_phrases = set()
    print "--extracting NP"
    noun_phrases = set([lemmatize(phrase) for phrase in extract_candidate_chunks(doc)])

    vectorizer = TfidfVectorizer(decode_error='ignore', preprocessor=preprocess, ngram_range=(1, 3), tokenizer=tokenize)
    analyzer = vectorizer.build_analyzer()
    phrases = list(set([phrase for phrase in analyzer(doc) if valid_ngram(phrase, noun_phrases)]))
    doc = preprocess(doc)

    tfidf = []
    first_occurrence = []
    entropy = []
    length = []
    doc_len = len(doc)

    entropy = get_entropy_doc(doc, phrases)
    # calculate tfidf
    for i, phrase in enumerate(phrases):
        first_occurrence.append(doc.find(phrase) / doc_len)
        tf = doc.count(phrase)
        if phrase in df_dic:
            tfidf[i] = tf * log10(training_size / df_dic[phrase])
        else:
            tfidf[i] = tf * log10(training_size)
        length.append(len(phrase))
    return phrases, tfidf, first_occurrence, entropy, length


# Takes as input a list of doc tokens (nested list)
# For instance, with two docs: [[token_1, token_2], [token_3, token_4, token_5]]
def extract_features_test(docs, keys):
    tfidf_matrix, phrase_list, first_occurrence_all, entropy_all = get_tfidf_matrix(docs)
    #entropy_all = [get_phrase_entropy(doc, phrase_list) for doc in docs]
    features_doc, labels_doc, phrase_idx_doc, phrase_list = get_candidates_for_docs(tfidf_matrix, phrase_list, keys, first_occurrence_all, entropy_all)
    return features_doc, labels_doc, phrase_idx_doc, phrase_list

# extract noun phrases from text
# implemented by Burton DeWilde, http://bdewilde.github.io/blog/2014/09/23/intro-to-automatic-keyphrase-extraction/
def extract_candidate_chunks(text, grammar=r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}'):
    import itertools, nltk, string

    # exclude candidates that are stop words or entirely punctuation
    punct = set(string.punctuation)
    # tokenize, POS-tag, and chunk using regular expressions
    chunker = nltk.chunk.regexp.RegexpParser(grammar)
    tagged_sents = nltk.pos_tag_sents(nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(text))
    all_chunks = list(itertools.chain.from_iterable(nltk.chunk.tree2conlltags(chunker.parse(tagged_sent))
                                                    for tagged_sent in tagged_sents))
    # join constituent chunk words into a single chunked phrase
    candidates = [' '.join(word for word, pos, chunk in group).lower()
                  for key, group in itertools.groupby(all_chunks, lambda (word,pos,chunk): chunk != 'O') if key]

    return [cand for cand in candidates \
            if cand not in stopwords and not all(char in punct for char in cand)]

# input doc string, chunk number of N
# output a list of evenly splited chunks
def split_doc_into_chunks(doc, N=10):
    chunks = []
    chunk_size = len(doc) // N
    for i in range(N - 1):
        chunks.append(doc[i * chunk_size:(i + 1) * chunk_size])
    chunks.append(doc[(N - 1) * chunk_size:])
    return chunks

# input a phrase, and doc in chunks, return entropy
def get_entropy(phrase, chunks):
    p = 0
    tf_c = []
    for chunk in chunks:
        tf_c.append(chunk.count(phrase))
    tf = sum(tf_c)
    if tf == 0:
        return 0
    else:
        for val in tf_c:
            if val!= 0:
                p += (-1) * (val / tf) * log10(val / tf)
        return p

# input: one doc, list of phrases
# output: list of entropy, in given order of phrases
def get_entropy_doc(doc, phrases, N=10):
    entropy = []
    # split doc into N chunks
    chunks = split_doc_into_chunks(doc, N)
    for phrase in phrases:
        p = 0
        tf = doc.count(phrase)
        if tf == 0:
            continue
        for chunk in chunks:
            tf_c = chunk.count(phrase)
            if tf_c != 0:
                p += (-1) * (tf_c / tf) * log10(tf_c / tf)
        entropy.append(p)
    return entropy

def get_phrase_entropy(doc, phrases, N=10):
    entropy = {}
    doc = preprocess(doc)
    # split doc into N chunks
    chunks = split_doc_into_chunks(doc, N)
    for phrase in phrases:
        p = 0
        tf = doc.count(phrase)
        if tf == 0:
            continue
        for chunk in chunks:
            tf_c = chunk.count(phrase)
            if tf_c != 0:
                p += (-1) * (tf_c / tf) * log10(tf_c / tf)
        entropy[phrase] = p
    return entropy

# remove ngrams that start and end with stopwords
def valid_ngram(ngram, noun_phrases):
    if len(ngram) <= 1:
        return False
    grams = ngram.split()
    if grams[0] in stopwords or grams[-1] in stopwords:
        return False
    # other heuristics for filtering go here...
    if noun_phrases:
        if ngram not in noun_phrases:
            return False
    return True

# learn vocabulary from list of docs and calculate first occurrence scores at the same time
# two kinds of criteria: noun phrase, heuristics
def learn_vocabulary(docs, only_noun_phrases=True):
    first_occurrence_all = []
    entropy_all = []
    docs = [doc.decode('utf8', 'ignore') for doc in docs]

    noun_phrases = set()
    if only_noun_phrases:
        for i, doc in enumerate(docs):
            print "--extracting NP from doc", i
            #doc = doc.decode('utf8', 'ignore')
            noun_phrases.update([lemmatize(phrase) for phrase in extract_candidate_chunks(doc)])
    # with open('./nlm500_test_docs_noun_phrases.set', 'w') as f:
    #     pickle.dump(noun_phrases, f)
    #
    # print "loading pre-extracted set of noun_phrases"
    # noun_phrases = set()
    # with open('./nlm500_test_docs_noun_phrases.set', 'r') as f:
    #     noun_phrases = pickle.load(f)

    vectorizer = TfidfVectorizer(decode_error='ignore', preprocessor=preprocess, ngram_range=(1, 3), tokenizer=tokenize)
    analyzer = vectorizer.build_analyzer()
    vocab = set()
    print "--learning vocabulary"
    for i, doc in enumerate(docs):
        print "--learning doc", i
        first_occurrence = {}
        entropy = {}

        phrases = analyzer(doc) # all phrases from doc
        doc = preprocess(doc) # do the same preprocessing to doc for locating phrase
        doc_length = len(doc)
        for i, phrase in enumerate(phrases):
            if valid_ngram(phrase, noun_phrases) and phrase not in first_occurrence:
                pos = doc.find(phrase)
                if pos == -1:
                    print "--phrase: '{}' not found".format(phrase)
                    continue
                first_occurrence[phrase] = pos / doc_length
                # calcualte entropy
                chunks = split_doc_into_chunks(doc)
                entropy[phrase] = get_entropy(phrase, chunks)
                vocab.add(phrase)
        first_occurrence_all.append(first_occurrence)
        entropy_all.append(entropy)
    print "--size of vocabulary: ", len(vocab)
    return vocab, first_occurrence_all, entropy_all

# input: list of docs as strings: ['doc 1 string', 'doc 2 string']
# output: tfidf matrix, each row a doc, each col a phrase, each cell a tfidf score
#         list of vocabulary in the same order as features
#         record of first occurrence of each valid ngram in each doc
def get_tfidf_matrix(docs):

    vocab, first_occurrence_all, entropy_all = learn_vocabulary(docs)
    vectorizer = TfidfVectorizer(decode_error='ignore', preprocessor=preprocess, ngram_range=(1, 3), tokenizer=tokenize)

    print "--transforming tfidf matrix"
    vectorizer.vocabulary = list(vocab)
    X = vectorizer.fit_transform(docs)
    # get list of phrases in the order of the feature vector
    vocab_list = [phrase for phrase, idx in sorted(vectorizer.vocabulary_.items(), key=operator.itemgetter(1))]
    assert(len(vocab_list) == X.shape[1])
    return X, vocab_list, first_occurrence_all, entropy_all
    #return preprocessor, tokenizer, analyze

# Input: parameters to determine features
# Ouput: feature vector for a single keyphrase of size len(features)
def get_feature_vector(phrase, tfidf, first_occurrence, entropy):
    #feature_vec = np.zeros((1, len(features)))
    feature_vec = []
    for f in features:
        if f == 'tfidf':
            feature_vec.append(tfidf)
        elif f == 'first_occurrence':
            feature_vec.append(first_occurrence)
        elif f == 'entropy':
            feature_vec.append(entropy)
        elif f == 'length':
            feature_vec.append(len(phrase))
        elif f == 'num_tokens':
            feature_vec.append(len(phrase.split()))
    return feature_vec

# input: tfidf_matrix, list of all phrases in vocab, set of all true keywords for each doc
# output: feature matrix (np.array): [[feature vector1], [feature vector2], ...], labels:[0, 1, ...]
def get_feature_matrix(tfidf_matrix, phrase_list, true_keys, first_occurrence, phrase_entropy):
    #X = np.empty((0, len(features)))
    #y = np.empty(0)
    X = []
    y = []
    doc_tfidf_vecs = tfidf_matrix.toarray().tolist() # tfidf matrix

    # lower true keywords
    true_keys = [[key.lower() for key in key_list] for key_list in true_keys]

    for doc_id, tfidf_vec in enumerate(doc_tfidf_vecs):
        # traverse the doc vector
        print "--extracting features from doc {}".format(doc_id)
        for i, tfidf in enumerate(tfidf_vec):
            if tfidf != 0: # Why is this case here?
                feature_vec = get_feature_vector(phrase_list[i], tfidf, first_occurrence[doc_id][phrase_list[i]], phrase_entropy[doc_id][phrase_list[i]])
                #X = np.append(X, feature_vec, axis=0)
                X.append(feature_vec)
                label = lambda: 1 if phrase_list[i] in true_keys[doc_id] else 0
                y.append(label())
                #y = np.append(y, label())

    return np.array(X), y

def get_candidates_for_docs(tfidf_matrix, phrase_list, true_keys, first_occurrence, entropy_all):
    doc_tfidf_vecs = tfidf_matrix.toarray().tolist() # tfidf matrix

    # lower true keywords
    true_keys = [[key.lower() for key in key_list] for key_list in true_keys]

    features_doc = []
    labels_doc = []
    phrase_idx_doc = []

    for doc_id, tfidf_vec in enumerate(doc_tfidf_vecs):
        #X = np.empty((0, len(features)))
        #y = np.empty(0)
        X = []
        y = []
        phrase_idx = []
        # traverse the doc vector
        print "--extracting features from doc {}".format(doc_id)
        for i, tfidf in enumerate(tfidf_vec):
            if tfidf != 0: # Why is this case here?
                feature_vec = get_feature_vector(phrase_list[i], tfidf, first_occurrence[doc_id][phrase_list[i]], entropy_all[doc_id][phrase_list[i]])
                #X = np.append(X, feature_vec, axis=0)
                X.append(feature_vec)
                label = lambda: 1 if phrase_list[i] in true_keys[doc_id] else 0
                #y = np.append(y, label())
                y.append(label())
                phrase_idx.append(i)
        features_doc.append(np.array(X))
        labels_doc.append(y)
        phrase_idx_doc.append(phrase_idx)
    return features_doc, labels_doc, phrase_idx_doc, phrase_list


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
