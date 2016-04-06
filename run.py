# Run as follows: `run.py method_name dataset_dir dataset_name`
# For example, `run.py nlm graph_closeness`
import sys
from svm import train_svm, test_svm
from graph_method import GraphMethod
from import_datasets import get_dataset
from preprocess import tokenize, lemmatize, stem, remove_stopwords
from feature_extraction import get_grams_indices, extract_features, get_vec_differences

valid_methods = set(['graph_closeness', 'svm'])
valid_datasets = set(['nlm', 'js'])

def tokenize(docs):
    for i in range(len(docs)):
        docs[i] = lemmatize(tokenize(stem(remove_stopwords(docs[i]))))
    return docs

def main():
    method_name, data_dir, dataset_name = sys.argv[1:] # Assign last three args to method, data_dir, dataset
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
        X_train, y_train = extract_features(train_docs, train_keys)
        X_test, y_test = extract_features(test_docs, test_keys)
        svm = train_svm(X_train, y_train)
        accuracy = test_svm(svm, X_test, y_test)
    elif method_name == 'svm_ranking':
        X_train_vec, y_train_vec = extract_features(train_docs, train_keys)
        X_train, y_train = get_vec_differences(X_train_vec, y_train_vec)
        X_test_vec, y_test_vec = extract_features(test_docs, test_keys)
        X_test, y_test = get_vec_differences(X_test_vec, y_test_vec)
        svm = train_svm(X_train, y_train)
        # The test_svm function needs to be replaced for this method
        # so it finds the diff. of test vectors, classifies those
        # differences, and ranks using those classifications
        accuracy = test_svm(svm, X_test, y_test)
    print accuracy

if __name__ == '__main__':
    main()
