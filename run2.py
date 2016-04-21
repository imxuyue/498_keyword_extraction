'''
run2 accept two command line arguments
first one specifies the method used to extract the keywords
second one can be directory containing the docs and corresponding keywords,
then program will return the precision and recall
second one can be a file, then program will output the keywords
'''
import os
import sys
from graph_method import GraphMethod
from tfd import TFD

VALID_METHODS = set(['tf', 'td', 'tfd', 'closeness', 'textrank'])
EXTRACTORS = {'tf':TFD, 'td':TFD, 'tfd': TFD, 'closeness' : GraphMethod, 'textrank' : GraphMethod}

def main():
	method, path = sys.argv[1:]
	if method not in VALID_METHODS:
		print "Please input valide method name"

	else:
		if os.path.isdir(path):
			extractor = EXTRACTORS[method](path)
			accuracy, recall = extractor.get_accuracy_recall2(method)
			print "accuracy: " + str(accuracy)
			print "recall: " + str(recall)
		elif os.path.isfile(path):
			extractor = EXTRACTORS[method]('./')
			keywords = extractor.get_keywords(path, method)
			print keywords
		else:
			print "please input <method> <data_dir or filename>"


if __name__ == "__main__":
    main()			



