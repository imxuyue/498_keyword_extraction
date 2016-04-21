In this project, we have implemented various keywords/keyphrases extaction methods, including supervised methods and unsupervised methods. Since these two category methods will accept different number of parameters, we have two run files:

for run.py:
Run as follows: `$ python run.py method_name dataset_dir dataset_name`
For example, `run.py nlm graph_closeness`
method_name: choose from 'NB', 'graph_closeness', 'text_rank', 'svm', 'svm_ranking'
dataset_dir: the actual data directory
dataset_name: choose from 'nlm', 'js', should match the data directory you mentioned in the previous argument


run2.py is for unsupervised methods
To run this program, two command line arguments are needed:
The first one is method name used to extract keywords, vaild method names are listed below:
tf : term frequency based extraction
td : term distribution based extraction
tfd : assemble term frequency and term distribution
closeness : closeness centrality based graph method
textrank  : textrank centrality based graph method

The second argument can be a directory or a filename
if it's directory (containg docs and corresponding keywords), the program will output the average accuracy and recall for all the files in the directory; if it's filename, the program will output the extracted keywords/keyphrases
