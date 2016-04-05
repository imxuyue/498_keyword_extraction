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
Class Solution is implemented to speed up the process to get accuracy from a lot of files by using multiprocessing programming
"""

import re
import os
import sys
import threading
from multiprocessing import Process, Queue, cpu_count
from collections import defaultdict, deque
import PorterStemmer as stem  


class Keyword(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.stemmer = stem.PorterStemmer()
        self.stopwords = self.get_stopwords()

    def get_stopwords(self):
        stopwords = dict()
        with open('stopwords.txt', 'r') as f:
            for word in f:
                stopwords[word.rstrip()] = 0
        return stopwords

    def get_sentence_list(self, filename):
        with open(self.data_dir + filename, 'r') as f:
            self.sentence_list = re.sub(r'[^a-z.?!-]', ' ', f.read())
            self.sentence_list = filter(lambda x: False if len(x) <= 1 else True, re.split(r'[.!?]', self.sentence_list))
            for index, sentence in enumerate(self.sentence_list[:]):
                self.sentence_list[index] = sentence.split()

    def remove_stopwords(self):
        for index, sentence in enumerate(self.sentence_list[:]):
            self.sentence_list[index] = filter(lambda word: False if word in self.stopwords else True, sentence)

    def stem_words(self):
        for index, sentence in enumerate(self.sentence_list[:]):
            self.sentence_list[index] = [self.stemmer.stem(word, 0, len(word) - 1) for word in sentence]

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
        self.get_items()

    def fit(self, filename, ans_file):
        print filename
        self.get_keywords(filename)
        self.get_ans(ans_file)
        
    def predict(self, filename):
        self.kw = self.get_keywords(filename)
        return self.kw

    def get_ans(self, ans_file):  #get the given keywords from the file
        self.ans = []
        with open(self.data_dir + ans_file, 'r') as f:
            self.ans = f.read().split()
            self.ans = set(filter(lambda a: a in self.items, self.ans))

    def get_accuracy_recall(self):
        hit = 0.0
        for word in self.kw:
            if word in self.ans:
                hit += 1
        print len(self.kw), len(self.ans)
        return hit/len(self.kw), hit/len(self.ans)


class Graph(object):
    def __init__(self, sentence_list, items):
        self.sentence_list = sentence_list
        self.items = items

    def build_co_occurrence(self):
        self.co_occurrence = dict()
        self.neighbors = defaultdict(list)
        for sentence in self.sentence_list:
            for w1 in sentence[:-1]:
                for w2 in sentence[1:]:
                    if w1 != w2:
                        self.co_occurrence[(w1, w2)] = self.co_occurrence.get((w1, w2), 0) + 1
                        self.neighbors[w1].append(w2)
                        self.neighbors[w2].append(w1)

    def build_undirected_weighted_edges(self):
        self.edges = dict()
        for w1 in self.items:
            for w2 in self.items:
                if w1 != w2:
                    if (w1, w2) in self.co_occurrence or (w2, w1) in self.co_occurrence:
                        self.edges[(w1, w2)] = self.co_occurrence.get((w1, w2), 0) + self.co_occurrence.get((w2, w1), 0)
                    else:
                        self.edges[(w1, w2)] = 0.001  #build a edge with small weight if two words do not co-occurrence

class Closeness(Keyword, Graph):
    def get_shortest_path(self, root):   #using the Dijkstra's algorithm to get the shortest path from root node to all the other nodes
        self.dist = dict()
        queue = dict()
        for item in self.items:
            self.dist[item] = 100000
            if item == root:
                self.dist[root] = 0
            queue[item] = self.dist[item]
        while queue:
            item = min(queue.items(), key = lambda item: item[1])[0]
            del queue[item]
            for key in self.neighbors[item]:
                temp = self.dist[item] + 1.0/self.edges[(item, key)]
                if temp < self.dist[key]:
                    self.dist[key] = temp
                    queue[key] = temp

    def get_keywords(self, filename):
        self.kw = []
        self.pre_processing(filename)
        self.build_co_occurrence()
        self.build_undirected_weighted_edges()
        n = len(self.items)
        score = dict()
        rank = len(self.items)/3   #get the top 1/3 as the potential keywords
        for item in self.items:
            self.get_shortest_path(item)
            score[item] = (n - 1) * 1.0 / sum(self.dist.values())
        temp = sorted(score.items(), key = lambda items: items[1], reverse = True) 
        self.kw = [item[0] for item in temp][:rank]


class TextRank(Keyword, Graph):
    def __init__(self, data_dir, d = 0.85, c = 1.0):
        Keyword.__init__(self, data_dir)
        self.d = d
        self.c = c

    def is_convergence(self, old_score, new_score, error = 0.001):
        temp = 0
        for key in old_score:
            temp += abs(old_score[key] - new_score[key])
        return temp <= error

    def get_keywords(self, filename):
        self.kw = []
        self.pre_processing(filename)
        self.build_co_occurrence()
        self.build_undirected_weighted_edges()
        old_score = dict()
        new_score = dict()
        rank = len(self.items) / 3 #use the top 1/3 words as the potential keywords
        for item in self.items:
            old_score[item] = self.c

        while True:
            for w1 in self.items:
                in_sum = 0
                for w2 in self.neighbors[w1]:
                    out_sum = sum(self.edges[(w2, w3)] for w3 in self.neighbors[w2] if w3 != w2)
                    if w2 != w1:
                        in_sum += (self.edges[(w1, w2)] * old_score[w2] / out_sum)
                new_score[w1] = 1 - self.d + self.d * in_sum

            if not self.is_convergence(old_score, new_score):
                old_score, new_score = new_score, dict()
            else:
                break
        temp = sorted(new_score.items(), key = lambda items: items[1], reverse = True)
        self.kw = [item[0] for item in temp][:rank]


class Solution(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.extractors = {'textrank': TextRank(data_dir), 'closeness' : Closeness(data_dir)}
        self.accuracy = {'textrank': 0.0, 'closeness' : 0.0}
        self.q_accuracy = {'textrank': Queue(), 'closeness': Queue()}
        self.filelist = filter(lambda filename: filename.endswith('txt'), os.listdir(self.data_dir))
        self.q_filelist = Queue()
        self.nfiles = len(self.filelist)
        self.ncores = cpu_count()

    def go_rank(self, key):
        extractor = self.extractors[key]
        while True:
            if self.q_filelist.empty():
                break
            filename = self.q_filelist.get()
            ans_file = filename[:-3]  + 'key'
            extractor.fit(filename, ans_file)
            self.q_accuracy[key].put(extractor.get_accuracy_recall()[0])

    def get_accuracy(self, key):
        for filename in self.filelist:
            self.q_filelist.put(filename)
        self.accuracy[key] = 0.0
        processes = []
        for _ in range(self.ncores):  #construct ncores of processes to run simutaneously
            p = Process(target = self.go_rank, args = (key,))
            p.start()
            processes.append(p)
        for process in processes:
            process.join()
        while not self.q_accuracy[key].empty():
            self.accuracy[key] += self.q_accuracy[key].get()
        self.accuracy[key] /= self.nfiles
        return self.accuracy[key] 

    def get_accurancy_from_closeness_rank(self):
        return self.get_accuracy('closeness')

    def get_accuracy_from_text_rank(self):
        return self.get_accuracy('textrank')


if __name__ == '__main__':
    data_dir = sys.argv[1]
    solution = Solution(data_dir)
    print solution.get_accuracy_from_text_rank()
    print solution.get_accurancy_from_closeness_rank()






