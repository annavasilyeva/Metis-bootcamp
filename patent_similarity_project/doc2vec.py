import gensim
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
import re
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn
import time

start_time = time.time()
count = 0
lablist = []
for i in range(len(abstract_tot)): #abstract_tot is the list of all the scraped abstracts
    lablist.append(count)
    count+=1
tokenizer  = RegexpTokenizer(r'\w+')
en_stop = get_stop_words('en')
p_stemmer = PorterStemmer()

texts = []
taggeddoc = []
for index, i in enumerate(abstract_tot):
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)
    stopped_tokens = [j for j in tokens if not j in en_stop]
    number_tokens = [re.sub(r'[\d]', ' ', k) for k in stopped_tokens]
    number_tokens = ' '.join(number_tokens).split()
    stemmed_tokens = [p_stemmer.stem(p) for p in number_tokens]
    texts.append(number_tokens)
    #td = TaggedDocument(gensim.utils.to_unicode(str.encode(' '.join(stemmed_tokens))).split(),str(index))
    #taggeddoc.append(td)

LabeledSentence = gensim.models.doc2vec.LabeledSentence
class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
       self.labels_list = labels_list
       self.doc_list = doc_list
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            #yield LabeledSentence(words=doc.split(),tags=[self.labels_list[idx]])
            yield LabeledSentence(words=doc,tags=[self.labels_list[idx]])

#it = LabeledLineSentence(claimcalc_tot, lablist)
it = LabeledLineSentence(texts, lablist)

#model = gensim.models.Doc2Vec(size=300, window=10, min_count=5, workers=11,alpha=0.025, min_alpha=0.025) # use fixed learning rate
model = gensim.models.Doc2Vec(dm = 1, size=100, min_count=10, workers=11,alpha=0.025, min_alpha=0.025) # use fixed learning rate
model.build_vocab(it)
for epoch in range(10):
    print("epoch is: " + str(epoch))
    model.train(it)
    model.alpha -= 0.002 # decrease the learning rate
    model.min_alpha = model.alpha # fix the learning rate
    model.train(it)

end_time = time.time()
print("time taken is" + str(end_time-start_time))

model.save('trained.model')
model.save_word2vec_format('trained.word2vec')

#plot word2vec example
w2v = gensim.models.Doc2Vec.load_word2vec_format('trained.word2vec')
words_np = []
words_label = []
for word in w2v.vocab.keys():
    words_np.append(w2v[word])
    words_label.append(word)
print('Added %s words. Shape %s'%(len(words_np),np.shape(words_np)))
 
pca = PCA(n_components=2)
pca.fit(words_np)
reduced= pca.transform(words_np)
plt.figure(figsize=(10, 10))

for index,vec in enumerate(reduced):
        # print ('%s %s'%(words_label[index],vec))
        if index <100:
            x,y=vec[0],vec[1]
            plt.scatter(x,y)
            plt.annotate(words_label[index],xy=(x,y), fontsize=14)
plt.axis([0,.2,-.5,.5])
plt.show()