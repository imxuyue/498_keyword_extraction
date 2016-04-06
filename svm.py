import numpy as np
from sklearn.svm import SVC

def train_svm(X, y):
    clf = SVC()
    clf.fit(X, y)
    return clf

def test_svm(clf, X, y):
    return clf.score(X, y)
