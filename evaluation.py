'''
evaluation.py
-------------
precision, recall, etc
'''
from __future__ import division
from feature_extraction import extract_features_test

def get_precision(true_keys, pred_keys):
    true_keys = set([key.lower() for key in true_keys])
    correct = len(true_keys.intersection(set(pred_keys)))
    print "pred keys: {} correct: {}".format(len(pred_keys), correct)
    if len(pred_keys) == 0:
        return 0
    return correct / len(pred_keys)

def get_recall(true_keys, pred_keys):
    true_keys = set([key.lower() for key in true_keys])
    correct = len(true_keys.intersection(set(pred_keys)))
    if len(pred_keys) == 0:
        return 0
    return correct / len(true_keys)

def evaluate_on_each_doc(classifier, features_doc, labels_doc, phrase_idx_doc, phrase_list, true_keys_doc):
    precisions = np.zeros(len(features_doc))
    recalls = np.zeros(len(features_doc))
    # go through each document
    for i in range(len(features_doc)):
        pred_labels = classifier.predict(feature_doc[i])
        pred_keys = []
        print "**pred keys: {}".format(sum(pred_labels))
        # collect all phrases that has pred label 1
        for label, idx in zip(pred_labels, phrase_idx_doc[i]):
            if label == 1:
                pred_keys.append(phrase_list[idx])

        precisions[i] = get_precision(true_keys_doc[i], pred_keys))
        recalls[i] = get_recall(true_keys_doc[i], pred_keys))

    return np.average(precisions), np.average(recalls)
