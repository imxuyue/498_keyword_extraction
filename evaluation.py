'''
evaluation.py
-------------
precision, recall, etc
'''
from __future__ import division
from feature_extraction import extract_features_test

def get_precision(true_keys, pred_keys):
    true_keys = [key.lower() for key in true_keys]
    correct = len(set(true_keys) & set(pred_keys))
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

def evaluate_on_each_doc(classifier, features_doc, labels_doc, phrase_idx_doc, phrase_list, true_keys_doc):
    precisions = []
    recalls = []
    # go through each document
    for features, labels, phrase_indices, true_keys in zip(features_doc, labels_doc, phrase_idx_doc, true_keys_doc):
        pred_labels = classifier.predict(features)
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
