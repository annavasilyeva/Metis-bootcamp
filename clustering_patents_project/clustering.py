import pandas as pd
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF

stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(['method','based','provided','includes','device','second',
                  'including','include','first','one','may','user','using',
                  'associated','methods','method'])
def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is taken care of
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

totalvocab_stemmed = []
totalvocab_tokenized = []
for i in abstract_list: # abstract_list should be list of all abstracts from scraping
    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)    
vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_tokenized)
print ('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')
# using high number of features
tfidf_vectorizer = TfidfVectorizer(max_df=0.9, max_features=200000,
                                 min_df=3, stop_words=stopwords, 
                                 use_idf=True, tokenizer=tokenize_only, ngram_range=(1,3))

tfidf_matrix = tfidf_vectorizer.fit_transform(abstract_list) #fit the vectorizer to abstract
print(tfidf_matrix.shape)

##########################  NMF  ##############################
n_topics = 6 
n_top_words = 10

def print_top_words(model, feature_names, n_top_words):
   for topic_idx, topic in enumerate(model.components_):
       print("Topic #%d:" % topic_idx)
       print(" ".join([feature_names[i]
                       for i in topic.argsort()[:-n_top_words - 1:-1]]))
   print()
nmf = NMF(n_components=n_topics, random_state=2, alpha=.1, l1_ratio=.5).fit(tfidf_matrix)
print("\nTopics in NMF model:")
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
print_top_words(nmf, tfidf_feature_names, n_top_words)
 
terms = tfidf_vectorizer.get_feature_names()

###################  K-means  ################################
num_clusters = 6 # determined using elbow method
km = KMeans(n_clusters=num_clusters,random_state=3425)
km.fit(tfidf_matrix)
clusters = km.labels_.tolist()

pat = {'title': title_list, 'company': company_list, 'abstract': abstract_list, 'docID': docID_list, 'date':date_list, 'cluster':clusters}
frame = pd.DataFrame(pat, index = [clusters] , columns = ['title', 'company', 'date', 'abstract','cluster'])
print(frame['cluster'].value_counts()) # number of patents per cluster (clusters from 0 to 6)
grouped = frame['company'].groupby(frame['cluster']) # groupby cluster for aggregation purposes

print("Top terms per cluster:")
print()
#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1] 

for i in range(num_clusters):   
    print("Cluster %d words:" % i, end='')    
    for ind in order_centroids[i, :10]: #replace 10 with n words per cluster
        print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
    print() 
    print()     
#    print("Cluster %d titles:" % i, end='')
#    for title in frame.ix[i]['title'].values.tolist():
#        print(' %s,' % title, end='')
#        print()
#    print() #add whitespace
#    print() #add whitespace
    
print()
print()

#grouping = frame.sort_values(["date","cluster","company"], ascending=True)
grouping = frame.groupby(["date","cluster","company"]).size()
def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')
print_full(grouping)