'''
evaluation.py
-------------
precision, recall, etc
'''
from __future__ import division
from feature_extraction import extract_features_test
import naive_bayes as NB

def get_precision(true_keys, pred_keys):
    true_keys = [key.lower() for key in true_keys]
    correct_keys = set(true_keys) & set(pred_keys)
    print "pred correct keys:"
    for key in list(correct_keys):
        print key
    correct = len(correct_keys)
    print "pred keys: {} correct: {}".format(len(pred_keys), correct)
    if len(pred_keys) == 0:
        return 0
    return correct / len(pred_keys)

def get_recall(true_keys, pred_keys):
    true_keys = [key.lower() for key in true_keys]
    correct = len(set(true_keys) & set(pred_keys))
    if len(pred_keys) == 0:
        return 0
    return correct / len(true_keys)

# N is the #keywords requested for each doc
def evaluate_on_each_doc(clf_name, clf, features_doc, labels_doc, phrase_idx_doc, phrase_list, true_keys_doc, N=10):
    precisions = []
    recalls = []
    # go through each document
    docid = 0
    for features, labels, phrase_indices, true_keys in zip(features_doc, labels_doc, phrase_idx_doc, true_keys_doc):
        print "*docid", docid
        docid += 1
        if clf_name == 'NB':
            pred_idx = NB.test(clf, N, features, labels)
            pred_keys = []
            # collect all phrases that has pred label 1
            for idx in pred_idx:
                pred_keys.append(phrase_list[phrase_indices[idx]])
            print "pred_keys:"
            print pred_keys
            print "true keys:"
            print true_keys
            precisions.append(get_precision(true_keys, pred_keys))
            recalls.append(get_recall(true_keys, pred_keys))

        if clf_name == 'svm':
            pred_labels = clf.predict(features, labels)
            pred_keys = []
            print "**pred keys: {}".format(sum(pred_labels))
            # collect all phrases that has pred label 1
            for label, idx in zip(pred_labels, phrase_indices):
                if label == 1:
                    pred_keys.append(phrase_list[idx])
            precisions.append(get_precision(true_keys, pred_keys))
            recalls.append(get_recall(true_keys, pred_keys))


    precision_avg = sum(precisions) / len(precisions)
    recall_avg = sum(recalls) / len(recalls)
    return precision_avg, recall_avg
