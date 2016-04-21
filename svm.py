import numpy as np
from sklearn.svm import SVC

def train_svm(X, y):
    clf = SVC(probability=True, class_weight='balanced')
    clf.fit(X, y)
    return clf

def test(clf, N, X_test):
    prob = clf.predict_log_proba(X_test)

    key_prob = prob[:, 1].tolist()
    #return key_prob
    sorted_idx = [i[0] for i in sorted(enumerate(key_prob), key=lambda x:x[1], reverse=True)]
    topN_idx = sorted_idx[:N]

    return topN_idx

def test_svm(clf, X, y):
    return clf.score(X, y)
