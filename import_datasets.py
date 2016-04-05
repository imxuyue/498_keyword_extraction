# Can import the functions using `from import_datasets import import_data_nlm` or similar

from os import listdir
from os.path import isfile, join, basename
import sys

def read_files(filenames):
    keys = [open(x).read().splitlines() for x in filenames]
    docs = [open(x[:-4] + '.txt').read().replace('\n', ' ').replace('\r', ' ') for x in filenames]
    return keys, docs

# Run using `docs = import_data_nlm(path)` where path = {path to the NLM_500 documents folder}
# This will give you a dictionary: `docs`
# The dict has the document ID as the key and a tuple/list of the [keys, doc] as the value
# For example, docs['122025511'] = [ [key1, key2, ...], 'document string' ]
def import_data_nlm(datasets_folder):
    all_filenames = [join(datasets_folder, f) for f in listdir(datasets_folder) if isfile(join(datasets_folder, f))]
    filenames = [x for x in all_filenames if x[-4:] == '.key']
    keys, texts = read_files(filenames)

    docs = {}
    for i in range(len(filenames)):
        file_id = basename(filenames[i])[:-4]
        docs[file_id] = (keys[i], texts[i])

    return docs

# Run using `train, test = import_data_js(path)` where path = {path to the data_js folder}
# This will give you two dictionaries: `train` and `test`
# Each dict has the document ID as the key and a tuple/list of the [keys, doc] as the value
# For example, train['C-41'] = [ [key1, key2, ...], 'document string' ]
def import_data_js(datasets_folder):
    train_path = join(datasets_folder, 'train_js/')
    test_path = join(datasets_folder, 'test_js/')
    train_filenames = [join(train_path, f) for f in listdir(train_path) if isfile(join(train_path, f))]
    test_filenames = [join(test_path, f) for f in listdir(test_path) if isfile(join(test_path, f))]
    train_names = [x for x in train_filenames if x[-4:] == '.key']
    test_names = [x for x in test_filenames if x[-4:] == '.key']
    train_keys, train_docs = read_files(train_names)
    test_keys, test_docs = read_files(test_names)

    train = {}
    for i in range(len(train_names)):
        train_id = basename(train_names[i])[:-4]
        train[train_id] = (train_keys[i], train_docs[i])
    test = {}
    for i in range(len(test_names)):
        test_id = basename(test_names[i])[:-4]
        test[test_id] = (test_keys[i], test_docs[i])

    return train, test

