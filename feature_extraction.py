import numpy as np

# Returns unigrams, bigrams, trigrams dicts
# For example, `trigrams[1] = ['house', 'is', 'nice']`
def get_grams_indices(tokens):
    unigrams, bigrams, trigrams = [], [], []
    for i in range(len(tokens)):
        if i < len(tokens)-2:
            trigrams[i] = tokens[i:i+2]
        if i < len(tokens)-1:
            bigrams[i] = tokens[i:i+1]
        unigrams[i] = tokens[i]
    return unigrams, bigrams, trigrams

# Takes as input a list of doc tokens (nested list)
# For instance, with two docs: [[token_1, token_2], [token_3, token_4, token_5]]
def extract_features(doc_tokens):
    for tokens in doc_tokens:
        unigrams, bigrams, trigrams = get_grams_indices(tokens)
        # Need to extract features (length of word, entropy, etc.)
    return features # Same length as doc_tokens (number of docs)
