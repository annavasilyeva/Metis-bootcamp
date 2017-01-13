import gensim
import re
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
import csv
import json

usertext = 'A method begins, decoding slice to produce a set, and obtaining a set. The method continues with the processing module generating a set, generating a set of blind passwords base, and generating a set of passkeys based. The method continues with the absorption generating a set based on the set and the set, decrypting the set to produce a set, and decoding the set to reproduce the data.'
title = "Retrieving information in a storage network (your title)"
# tokenize the new text
tokenizer  = RegexpTokenizer(r'\w+')
en_stop = get_stop_words('en')
tokens = tokenizer.tokenize(usertext.lower())
stopped_tokens = [j for j in tokens if not j in en_stop]
number_tokens = [re.sub(r'[\d]', ' ', k) for k in stopped_tokens]
number_tokens = ' '.join(number_tokens).split()
text = number_tokens

model = gensim.models.Doc2Vec.load('trained.model')
new_doc_vec = model.infer_vector(text)
best = model.docvecs.most_similar([new_doc_vec])

source_list = []
target_list = []
weight_list = []
num_list = []
column1 = []
column2 = []
column3 = []
weight = []
url = []

for i in range(len(best)):
    if best[i][1] > .45: # don't want to include any doc thats really not similar
        best2 = model.docvecs.most_similar(int(best[i][0])) 
        for j in range(len(best2)):
            if best2[j][1] > .5:
                best3 = model.docvecs.most_similar(int(best2[j][0]))
                for k in range(len(best3)):
                    if best3[k][1] > .5:
                        #cut of title after character 85 due to current d3 limitation
                        child1 = title_tot[best[i][0]]+' (US' + docID_tot[best[i][0]][1:]+', '+date_tot[best[i][0]][:4]+')'
                        if len(child1) > 85:
                            child1 = child1[:85]
                        child2 = title_tot[best2[j][0]]+' (US' + docID_tot[best2[j][0]][1:]+', '+date_tot[best2[j][0]][:4]+')'
                        if len(child2) > 85:
                            child2 = child2[:85]
                        child3 = title_tot[best3[k][0]]+' (US' + docID_tot[best3[k][0]][1:]+', '+date_tot[best3[k][0]][:4]+')'
                        if len(child3) > 85:
                            child3 = child3[:85]
                        column1.append(child1)
                        column2.append(child2)
                        column3.append(child3)
                        weight.append(best3[k][1])
                        urlstring = "https://patents.google.com/patent/US"+docID_tot[int(best3[k][0])][1:] 
                        url.append(urlstring)
with open('testjson.csv', 'w',newline='') as f:
    writer = csv.writer(f)
    writer.writerows(zip(column1,column2,column3,weight,url))

class Node(object):
    def __init__(self, name, size=None, url=None):
        self.name = name
        self.children = []
        self.size = size
        self.url = url

    def child(self, cname, size=None, url=None):
        child_found = [c for c in self.children if c.name == cname]
        if not child_found:
            _child = Node(cname, size, url)
            self.children.append(_child)
        else:
            _child = child_found[0]
        return _child

    def as_dict(self):
        res = {'name': self.name}
        if self.size is None or self.url is None:
            res['children'] = [c.as_dict() for c in self.children]
        else:
            res['size'] = self.size
            res['url'] = self.url
        return res

root = Node('Retrieving information in a storage network')

with open('testjson.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        grp1,grp2,grp3,size,url = row
        root.child(grp1).child(grp2).child(grp3, size,url)

outfile = open("out.json","w")
outfile.write(json.dumps(root.as_dict(),indent=4))
outfile.close()
print (json.dumps(root.as_dict(), indent=4))
    
    
