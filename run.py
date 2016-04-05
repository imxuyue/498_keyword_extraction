# Run as follows: `run.py method_name dataset_dir dataset_name`
# For example, `run.py nlm graph_closeness`
import sys, random
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
        random_test_indices = random.sample(range(len(docs)), len(docs)/10) # Use 10% as test
        train, test = [], []
        for i in range(len(docs)):
            if i in random_test_indices:
                test.append(docs[i])
            else:
                train.append(docs[i])
    return train, test

def main():
    method_name, data_dir, dataset_name = sys.argv[1:] # Assign last three args to method, dataset
    if (method_name not in valid_methods) or (dataset_name not in valid_datasets):
        print 'Invalid arguments, exiting!'
        sys.exit()
    train, test = get_dataset(data_dir, dataset_name)

    if method_name == 'graph_closeness':
        graph_method = GraphMethod(train, test)
        accuracy = graph_method.get_accuracy_from_closeness_rank()
    elif method_name == 'text_rank':
        graph_method = GraphMethod(train, test)
        accuracy = graph_method.get_accuracy_from_text_rank()
    elif method_name == 'svm':
        train_vec, test_vec = extract_features(train), extract_features(test)
        svm = train_svm(train_vec)
        accuracy = test_svm(svm, test_vec)

    print accuracy

if __name__ == '__main__':
    main()
