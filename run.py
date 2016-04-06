# Run as follows: `run.py method_name dataset_dir dataset_name`
# For example, `run.py nlm graph_closeness`
import sys, random
import numpy as np
from import_datasets import import_data_nlm, import_data_js
from preprocess import tokenize, lemmatize, stem, remove_stopwords
from graph_method import GraphMethod

valid_methods = set(['graph_closeness', 'svm'])
valid_datasets = set(['nlm', 'js'])

def get_dataset(data_dir, dataset_name):
    if dataset_name == 'js':
        train, test = import_data_js(data_dir)
    if dataset_name == 'nlm':
        docs = import_data_nlm(data_dir)
        keys = docs.keys()
        random_test_indices = random.sample(range(len(docs)), len(docs)/10) # Use 10% as test
        train, test = {}, {}
        for k in keys:
            if i in random_test_indices:
                test[k] = docs[k]
            else:
                train[k] = docs[k]
    return train, test

def tokenize(docs):
    for i in range(len(docs)):
        docs[i] = lemmatize(tokenize(stem(remove_stopwords(docs[i]))))
    return docs

def main():
    method_name, data_dir, dataset_name = sys.argv[1:] # Assign last three args to method, dataset
    if (method_name not in valid_methods) or (dataset_name not in valid_datasets):
        print 'Invalid arguments, exiting!'
        sys.exit()

    train, test = get_dataset(data_dir, dataset_name)
    train_keys, train_docs = zip(*train.values())
    test_keys, test_docs = zip(*test.values())
    train_docs, test_docs = tokenize(train_docs), tokenize(test_docs)

    if method_name == 'graph_closeness':
        graph_method = GraphMethod(data_dir + '/' + dataset_name + '/')
        accuracy, recall = graph_method.get_accuracy_from_closeness_rank()
    elif method_name == 'text_rank':
        graph_method = GraphMethod(data_dir + '/' + dataset_name + '/')
        accuracy, recall = graph_method.get_accuracy_from_text_rank()
    elif method_name == 'svm':
        # Feature vectors
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
        svm = train_svm(train_vec)
        accuracy = test_svm(svm, test_vec)

    print accuracy

if __name__ == '__main__':
    main()
