import urllib.request, urllib.error, urllib.parse, os, zipfile
from lxml import etree
from bs4 import BeautifulSoup
import pickle
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def xmlSplitter(data,separator=lambda x: x.startswith(b'<?xml')):
  buff = []
  for line in data:
    if separator(line):
      if buff:
        yield b''.join(buff)
        buff[:] = []
    buff.append(line)
  yield b''.join(buff)

def first(seq,default=None):
  """Return the first item from sequence, seq or the default(None) value"""
  for item in seq:
    return item
  return default

alldates = ['ipg160216.zip','ipg160223.zip','ipg160301.zip'] # this will only read 3 weeks of data
allcomp = ["amazon", "google", "microsoft"]
abstract_list = []
company_list = []
title_list = []
docID_list = []
date_list=[]
count_all = 0
for i in range(len(alldates)):
    count_all = count_all+1
    print("im in week:" + str(count_all))
    datasrc="https://bulkdata.uspto.gov/data2/patent/grant/redbook/fulltext/2016/" + alldates[i]
    filename = datasrc.split('/')[-1]
    if not os.path.exists(filename):
        with open(filename,'wb') as file_write:
            r = urllib.request.urlopen(datasrc)
            file_write.write(r.read())

    zf = zipfile.ZipFile(filename)
    xml_file = first([ x for x in zf.namelist() if x.endswith('.xml')])
    assert xml_file is not None

    count = 0
    count_abs = 0

    for item in xmlSplitter(zf.open(xml_file)):
        count += 1
        #if count > 10: break
        bs = BeautifulSoup(item)
        patent_util = bs.findAll('application-reference',{'appl-type':'utility'})
        if len(patent_util) != 0: 
            doc = etree.XML(item)
            date = "-".join(doc.xpath('//publication-reference/document-id/date/text()'))
            docID = first(doc.xpath('//publication-reference/document-id/doc-number/text()'))
            title = first(doc.xpath('//invention-title/text()'))
            assignee = first(doc.xpath('//assignee/addressbook/orgname/text()'))
            abstract = first(doc.xpath('//abstract/*/text()'))
            if(assignee != None):
                if any(word in assignee.lower() for word in allcomp):        
                    if abstract != None and len(abstract) > 100: # some abstracts get cut off          
                          count_abs += 1
                          abstract_list.append(abstract)
                          company_list.append(assignee)
                          title_list.append(title)
                          docID_list.append(docID)
                          date_list.append(date)

with open("abstract_16.txt","wb") as pickle_file:
    pickle.dump(abstract_list, pickle_file)
with open("company_16.txt","wb") as pickle_file2:
    pickle.dump(company_list,pickle_file2)
with open("title_16.txt","wb") as pickle_file3:
    pickle.dump(title_list,pickle_file3)
with open("docID_16.txt","wb") as pickle_file4:
    pickle.dump(docID_list,pickle_file4)
with open("date_16.txt","wb") as pickle_file5:
    pickle.dump(date_list,pickle_file5)