import numpy as np

def get_grams_indices(tokens):
    unigrams, bigrams, trigrams = [], [], []
    for i in range(len(tokens)):
        if i < len(tokens)-2:
            trigrams[i] = tokens[i:i+2]
        if i < len(tokens)-1:
            bigrams[i] = tokens[i:i+1]
        unigrams[i] = tokens[i]
    return unigrams, bigrams, trigrams

def extract_features(tokens):
    unigrams, bigrams, trigrams = get_grams_indices(tokens)
    # Need to extract features (length of word, entropy, etc.)
