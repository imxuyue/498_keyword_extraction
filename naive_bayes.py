'''
naive_bayes.py
--------------
a binary naive bayes classifier
'''
from __future__ import division
from sklearn.naive_bayes import GaussianNB
import numpy as np
def train(X_train, y_train):
    # Naive Bayes

    clf = GaussianNB()
    clf.fit(X_train, y_train)
    return clf
'''
def test(clf, X_test, y_test):
    prob = clf.predict_log_proba(X_test)

    key_prob = prob[:, 1].tolist()
    #return key_prob
    sorted_idx = [i[0] for i in sorted(enumerate(key_prob), key=lambda x:x[1], reverse=True)]
    N = 10
    top = sorted_idx[:N]
    precision = 0
    for idx in top:
        if y_test[idx] == 1:
            precision += 1
    precision /= N
    print "precision {}".format(precision)
'''


