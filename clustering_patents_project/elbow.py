import numpy as np
import pandas as pd
import csv
from scipy.spatial.distance import cdist
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from sklearn.cluster import KMeans

stopwords = nltk.corpus.stopwords.words('english')

stopwords.extend(['method','based','provided','includes','device','second',
                  'including','include','first','one','may','user','using',
                  'associated','methods','method'])
                  
def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

tfidf_vectorizer = TfidfVectorizer(max_df=0.9, max_features=200000,
                                 min_df=3, stop_words=stopwords, 
                                 use_idf=True, tokenizer=tokenize_only, ngram_range=(1,3))

tfidf_matrix = tfidf_vectorizer.fit_transform(abstract_list) #fit the vectorizer to abstract

k_range = range(1,20)
k_means_var = [KMeans(n_clusters=k).fit(tfidf_matrix) for k in k_range]
centroids = [X.cluster_centers_ for X in k_means_var]
k_euclid = [cdist(tfidf_matrix, cent, 'euclidean') for cent in centroids]
dist = [np.min(ke,axis=1) for ke in k_euclid]
wcss = [sum(d**2) for d in dist]

dfwcss = pd.DataFrame(wcss)
dataset = 'WCSS.csv'
dfwcss.to_csv(dataset, quotechar='\"', quoting=csv.QUOTE_NONNUMERIC,delimiter=',')
                            