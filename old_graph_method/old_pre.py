"""
extract the abstract from the text file 
accept two command line argumens:
1. input directory where the original text files are 
2. output directory where the extracted abstract will be 
"""

import os
import sys
import re
import PorterStemmer as stem  


input_dir = sys.argv[1]
output_dir = sys.argv[2]

ab_begin_marker = 'abstract'
ab_end_marker = 'categories and subject descriptors'


stopwords = dict()

with open('stopwords.txt', 'r') as f:
    for word in f:
        stopwords[word.rstrip()] = 0

for filename in os.listdir(input_dir):
    flag = 0
    with open(output_dir + filename, 'w+') as write_file:
        with open(input_dir + filename, 'r') as read_file:
            if filename.endswith('txt'):
                for line in read_file:
                    line = line.lower()
                    if line.startswith(ab_end_marker):
                        break
                    if flag:
                        write_file.write(line)
                    if line.startswith(ab_begin_marker):
                        flag = 1
            if filename.endswith('key'):
                words = read_file.read().lower().split()
                stemmer = stem.PorterStemmer()
                for word in words:
                    if word not in stopwords:
                        word = stemmer.stem(word, 0, len(word) - 1)
                        write_file.write(word + '\n')
