"""
By Jianming Sang 04/2016


Main idea:
    Graph based unsupervised algorithm for keywords extraction
    To build the graph, use terms as vertex and co-occurrence relation between terms as edges
    Use Closeness centrality and TextRank centrality to rank the vertex(terms), respectively
    Pick the k terms with highest centrality values as the key words 

Work flow:
    Extract sentences from the given text files and split each sentence into tokens
    Pre-process the tokens in each sentence: rmeove stopwords, stem the words
    Build the co-occurence matrix based on whether or not two terms appear in the same sentence
    Construct the undirected weighted graph using the tokens  as vertex and co-occurence as edge weight
    Calculate the centrality values based on different protocols for each vertex in the graph
    Extrac k tokens with highest centrality value as the key words
    Calcualte the accuracy and recall using the given keywords and the extracted keywords

To extract keywords, class Closeness or class TextRank can be used 
Class GraphMethod is implemented to speed up the calculation using multiprocessing programming
"""

import re
import os
import sys
import pandas as pd
import numpy as np  
from multiprocessing import Process, Queue, cpu_count
from collections import defaultdict, deque
import nltk
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


class Keyword(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.stopwords = stopwords.words('english')
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
    
    #split file conetent into each sentences for extracting co-occurrence 
    def get_sentence_list(self, filename):
        with open(self.data_dir + filename, 'r') as f:
            self.sentence_list = re.sub(r'[^a-z.?!-]+', ' ', f.read())
            self.sentence_list = filter(lambda x: False if len(x) <= 1 else True, re.split(r'[.!?]', self.sentence_list))
            for index, sentence in enumerate(self.sentence_list[:]):
                self.sentence_list[index] = sentence.split()

    def remove_stopwords(self):
        for index, sentence in enumerate(self.sentence_list[:]):
            self.sentence_list[index] = filter(lambda word: False if word in self.stopwords else True, sentence)

    def stem_words(self):
        for index, sentence in enumerate(self.sentence_list[:]):
            self.sentence_list[index] = [self.stemmer.stem(word) for word in sentence]

    def lemmatize(self):
        for index, sentence in enumerate(self.sentence_list[:]):
            self.sentence_list[index] = [self.lemmatizer.lemmatize(word) for word in sentence]
    
    #construct 2-gram, 3-gram for keyphrases extraction
    def construct_ngram(self):
        bigram, trigram = [], []
        for index, sentence in enumerate(self.sentence_list[:]):
            bigram = [' '.join(sentence[i:i+2]) for i in xrange(len(sentence) - 2)]
            trigram = [' '.join(sentence[i:i+2]) for i in xrange(len(sentence) - 3)]
            self.sentence_list[index].extend(bigram + trigram)
            self.sentence_list[index] = set(self.sentence_list[index])

    #get all items from each sentences
    def get_items(self):
        self.items = []
        for sentence in self.sentence_list:
            for word in sentence:
                self.items.append(word)
        self.items = list(set(self.items))

    def pre_processing(self, filename):
        self.get_sentence_list(filename)
        self.remove_stopwords()
        self.stem_words()
        self.lemmatize()
        self.construct_ngram()
        self.get_items()

    def get_keywords(self, filename):
        return

    def fit(self, filename, ans_file):
        self.get_keywords(filename)
        self.get_ans(ans_file)
        
    def predict(self, filename):
        self.kw = self.get_keywords(filename)
        return self.kw

    #get the given keywords from the file, process the keywords similarly to the main documents
    def get_ans(self, ans_file):  
        self.ans = []
        with open(self.data_dir + ans_file, 'r') as f:
            self.ans = f.read().split()
            self.ans = set(filter(lambda a: a in self.items, self.ans))

    # def get_ans(self, ans_file):  
    #     self.ans = []
    #     with open(self.data_dir + ans_file, 'r') as f:
    #         for line in f.readlines():
    #             line = re.split(r'[^a-z-]+', line.rstrip().lower())
    #             line = [word for word in line if word not in self.stopwords]
    #             line = [self.stemmer.stem(word) for word in line]
    #             line = [self.lemmatizer.lemmatize(word) for word in line]
    #             line = ' '.join(line)
    #             self.ans.append(line)
    #         self.ans = set(filter(lambda a: a in self.items, self.ans))

    def get_accuracy_recall(self):
        hit = 0.0
        for word in self.kw:
            if word in self.ans:
                hit += 1
        # print hit/len(self.kw), hit/len(self.ans)
        return hit/len(self.kw), hit/len(self.ans)


class Graph(object):
    def __init__(self, sentence_list, items):
        self.sentence_list = sentence_list
        self.items = items

    def build_co_occurrence(self):
        '''
        build co_occurrence relationship based on whether or not if two
        terms appear in the same sentence
        if word1 and word2 appear in the same sentence, add word1 to word2's
        neighbor list and word2 to word1's neighbor list
        ''' 
        self.co_occurrence = dict()
        self.neighbors = defaultdict(set)
        for sentence in self.sentence_list:
            for w1 in sentence:
                for w2 in sentence:
                    if w1 != w2:
                        self.co_occurrence[(w1, w2)] = self.co_occurrence.get((w1, w2), 0) + 1
                        self.co_occurrence[(w2, w1)] = self.co_occurrence.get((w2, w1), 0) + 1
                        self.neighbors[w1].add(w2)
                        self.neighbors[w2].add(w1)
    
    def build_undirected_weighted_edges(self):
        '''
        use the co_occurrence frequence as the edge weight
        if no co_occurrence, assign the edge a very small weigth (0.001 here)
        '''
        self.edges = dict()
        for w1 in self.items:
            for w2 in self.items:
                if w1 != w2:
                        self.edges[(w1, w2)] = self.co_occurrence.get((w1, w2), 0.001)

class Closeness(Keyword, Graph):
    def __init__(self, data_dir, rank):
        Keyword.__init__(self, data_dir)
        self.rank = rank

    def get_shortest_path(self):
        '''
        use Floyd-Warshall algorithm for calculating all pairs shortest path
        this step is the bottleneck for the closeness ranking method, which seems no way to solve
        '''
        self.dist = dict()
        for w1 in self.items:
            for w2 in self.items:
                if w1 == w2:
                    self.dist[(w1, w2)] = 0
                else:
                    self.dist[(w1, w2)] = 1.0 / self.edges[(w1, w2)]
        for k in self.items:
            for i in self.items:
                for j in self.items:
                    if self.dist[(i, j)] > self.dist[(i, k)] + self.dist[(k, j)]:
                        self.dist[(i, j)] = self.dist[(i, k)] + self.dist[(k, j)]

    def get_keywords(self, filename):
        self.pre_processing(filename)
        self.build_co_occurrence()
        self.build_undirected_weighted_edges()
        self.get_shortest_path()
        score = dict()
        rank = len(self.items) / self.rank  # NO. of extracted keywords, variable
        n = len(self.items)
        #closeness centrality of node i: (niterms - ) / sum(shortest path from node i to all other nodes)
        for item in self.items:
            score[item] = (n - 1) * 1.0 / sum(self.dist[(item, w2)] for w2 in self.items)

        temp = sorted(score.items(), key = lambda items: items[1], reverse = True) 
        self.kw = [item[0] for item in temp][:rank]
        return self.kw


class TextRank(Keyword, Graph):
    def __init__(self, data_dir, rank = 36, d = 0.85, c = 1.0):
        Keyword.__init__(self, data_dir)
        self.d = d
        self.c = c
        self.alpha = 0.1
        self.rank = rank
   
    def build_prob_matrix(self):
        '''
        build the transition probability matrix
        prob(i, j) denotes the probality jumping from item i to item j
        construct process:
        prob(i, j) = self.co_occurrence.get((i, j), 1 / niterms)
        prob(i, j) = prob(i, j) / sum(prob(i, :)) ---nomalize each entry use row sum
        prob(i, j) = (1 - alpha) * prob(i, j) + alpha / nitems (alpha is a hyperparameter)
        the last step is add a random jum probality in case of dead end 
        alpha is usually 0.1
        '''
        nitems = len(self.items)
        self.prob = pd.DataFrame(np.zeros((nitems, nitems)), columns=self.items, index=self.items)
        for w1 in self.items:
            for w2 in self.neighbors[w1]:
                self.prob.loc[w1, w2] = self.co_occurrence[(w1, w2)]
        self.prob = self.prob.div(self.prob.sum(axis=1), axis=0)
        self.prob = self.prob.fillna(1.0/nitems)
        self.prob = self.prob * (1 - self.alpha) + self.alpha / nitems

    def get_principle_left_eigenvector(self):
        '''
        calculat the principle left eigen vector as the corresponding steady probability that each
        term would be visited
        with higher probability, the term is more likely to be visited, which means this term is more important
        '''
        _, self.lvector = np.linalg.eig(self.prob.values.T)
        self.lvector = self.lvector[:, 0].T
        self.lvector /= sum(self.lvector)

    def get_keywords(self, filename):
        self.pre_processing(filename)
        self.build_co_occurrence()
        self.build_prob_matrix()
        self.get_principle_left_eigenvector()
        score = dict()
        for item, value in zip(self.items, self.lvector):
            score[item] = value  #score the term using the probability values
        rank = len(self.items) / self.rank  #again rank is a hyper parameter
        self.kw = [items[0] for items in sorted(score.items(), key = lambda item: item[1], reverse=True)[:rank]]
        return self.kw


class GraphMethod(object):
    def __init__(self, data_dir, rank = 36):
        self.data_dir = data_dir
        self.extractors = {'textrank': TextRank(data_dir, rank), 'closeness' : Closeness(data_dir, rank)}
        self.accuracy = {'textrank': 0.0, 'closeness' : 0.0}
        self.recall = {'textrank': 0.0, 'closeness' : 0.0}
        #Queue is thread safe
        self.q_accuracy = {'textrank': Queue(), 'closeness': Queue()}  #store accuracy for each file in multiprocess calculation
        self.q_recall = {'textrank': Queue(), 'closeness': Queue()} #store recall for each file in multiprocess calculation
        self.filelist = filter(lambda filename: filename.endswith('txt'), os.listdir(self.data_dir))
        self.q_filelist = Queue()  #word list in multiprocess programming
        self.nfiles = len(self.filelist)
        self.ncores = cpu_count()

    def go_rank(self, method):
        '''
        worker function in multiprocess programming
        '''
        extractor = self.extractors[method]
        while True:
            if self.q_filelist.empty():
                break
            filename = self.q_filelist.get()
            ans_file = filename[:-3]  + 'key'
            extractor.fit(filename, ans_file)
            accuracy, recall = extractor.get_accuracy_recall()
            self.q_accuracy[method].put(accuracy)
            self.q_recall[method].put(recall)

    #given the filename and method, return the keywords
    def get_keywords(self, filename, method):
        return self.extractors[method].get_keywords(filename)

    #return the precision and recall for the files in the given data dir using the given method
    def get_accuracy_recall2(self, method):
        for filename in self.filelist:
            self.q_filelist.put(filename)
        self.accuracy[method] = 0.0
        processes = []
        for _ in range(self.ncores):  #construct ncores of processes to run simutaneously
            p = Process(target = self.go_rank, args = (method,))
            p.start()
            processes.append(p)
        for process in processes:
            process.join()
        while not self.q_accuracy[method].empty():
            self.accuracy[method] += self.q_accuracy[method].get()
            self.recall[method] += self.q_recall[method].get()
        self.accuracy[method] /= self.nfiles
        self.recall[method] /= self.nfiles
        return self.accuracy[method], self.recall[method] 

    # def get_results_from_closeness_rank(self):
    #     return self.get_accuracy_recall2('closeness')

    # def get_results_from_text_rank(self):
    #     return self.get_accuracy_recall2('textrank')

    # def get_all_results(self):
    #     return [self.get_results_from_closeness_rank(), self.get_results_from_text_rank()]


if __name__ == '__main__':
    method = sys.argv[1]
    path = sys.argv[2]
    if os.path.isdir(path):
        graph = GraphMethod(path)
        accuracy, recall = graph.get_accuracy_recall2(method)
        print "accuracy: " + str(accuracy)
        print "recall: " + str(recall)
    elif os.path.isfile(path):

        graph = GraphMethod('/')
        keywords = graph.get_keywords(path, method)
        print keywords

    else:
        print "please input valid method name and data directory or filename"


    # data_dir = sys.argv[2]
    # dir2 = ['data/train_js/', 'data/train2_js/']
    # d = dir2[0]
    # with open('result2.txt', 'w') as f:
    #     for rank in [1, 3, 6, 9, 18, 36, 72]:
    #         f.write('rank = ' + str(rank) + '\n')
    #         print rank
    #         solution = GraphMethod(d, rank)
    #         accuracy1, recall1 = solution.get_results_from_text_rank()
    #         accuracy2, recall2 = solution.get_results_from_closeness_rank()
    #         f.write(d + ': ' + 'textrank: (' + str(accuracy1) + ', ' + str(recall1) + ')   ' \
    #                 'closeness: (' + str(accuracy2) + ', ' + str(recall2) + ')\n')
    #         print d + ': ' + 'textrank: (' + str(accuracy1) + ', ' + str(recall1) + ')   ' \
    #                 'closeness: (' + str(accuracy2) + ', ' + str(recall2) + ')\n'




