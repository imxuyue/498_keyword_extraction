import numpy as np

def construct_feature_vectors():
    train_data = np.array()
    test_data = np.array()
    for i in range(len(train_docs)):
        # Dict of {index of gram : list of words in gram}
        train_grams = get_grams(train_docs[i])
        for g in train_grams:
            if g in train_keys: # train_keys needs to be normalized I think
                is_keyword = True
            else:
                is_keyword = False
            train_data.append(extract_features(g, train_grams, is_keyword))
    for i in range(len(test_docs)):
        # Dict of {index of gram : list of words in gram}
        test_grams = get_grams(test_docs[i])
        for g in test_grams:
            if g in test_keys: # test_keys needs to be normalized I think
                is_keyword = True
            else:
                is_keyword = False
            test_data.append(extract_features(g, test_grams, is_keyword))
    # train_vec should probably be the first element of test_data, and the label
    # can be the second element of test_data. both are output by extract_features
    # above in the loop

def train_svm(X, y):
    clf = 0.0
    return clf

def test_svm(clf, X, y):
    accuracy = 0.0
    return accuracy
