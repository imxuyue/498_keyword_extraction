* Split files into training/testing set
* Tokenize to 1, 2 and 3  grams. All lower case, remove words appear too little times
* Take out grams that start&end in stopwords
* Lemmanization
* Construct feature vector
    1. TF-IDF
    2. Max number of letters
    3. Min number of words
    4. Entropy of distribution
    5. If it is in title
    6. Max word frequency
    7. Min word frequency
    8. First occurance (position in the document)
* Construct Importance-Comparasion Vector (ICV)
* Build ICV-label based on trainging data
* Train the model in SVM
* Run SVM on test data
* Convert imporatnce comparsion label into key phrase ranking
