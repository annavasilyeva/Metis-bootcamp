import urllib.request, urllib.error, urllib.parse, os, zipfile
from lxml import etree
from bs4 import BeautifulSoup
import pickle
import warnings
import time

warnings.filterwarnings("ignore", category=DeprecationWarning)

start_time = time.time()

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

# modify alldates to reflect the zip files you want read off uspto.gov 
alldates = ['ipg160112.zip',  'ipg160119.zip',  'ipg160126.zip',  'ipg160202.zip',  'ipg160209.zip', 
            'ipg160216.zip',  'ipg160223.zip',  'ipg160301.zip',  'ipg160308.zip',  'ipg160315.zip',  'ipg160322.zip',
            'ipg160329.zip',  'ipg160405.zip',  'ipg160412.zip',  'ipg160419.zip',  'ipg160426.zip',  'ipg160503.zip'] 

abstract_list = []
company_list = []
title_list = []
docID_list = []
date_list=[]
claim_list=[]
background_list = []
citation_list = []
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
    for item in xmlSplitter(zf.open(xml_file)):
        bs = BeautifulSoup(item)
        patent_util = bs.findAll('application-reference',{'appl-type':'utility'})
        doc = etree.XML(item)
        section = first(doc.xpath('//section/text()'))
        topclass = first(doc.xpath('//class/text()'))
        if len(patent_util) != 0 and str(section) == 'G'and str(topclass) == '06':              
            if count >= 3: break 
            count += 1
            print(count)
            item = item.replace(b'<?detailed-description description="Detailed Description" end="lead"?>',b'<detailed-description description="Detailed Description">')
            item = item.replace(b'<?detailed-description description="Detailed Description" end="tail"?>',b'</detailed-description>')
            item = item.replace(b'<?summary-of-invention description="Summary of Invention" end="lead"?>',b'<summary-of-invention description="Summary of Invention">')
            item = item.replace(b'<?summary-of-invention description="Summary of Invention" end="tail"?>',b'</summary-of-invention>')
            doc = etree.XML(item)
            date = "-".join(doc.xpath('//publication-reference/document-id/date/text()'))
            docID = first(doc.xpath('//publication-reference/document-id/doc-number/text()'))
            title = first(doc.xpath('//invention-title/text()'))
            assignee = first(doc.xpath('//assignee/addressbook/orgname/text()'))
            abstract = first(doc.xpath('//abstract/*/text()'))  
            summary = doc.xpath('//claim/claim-text/text()')  
            summary = [ii.strip('\n') for ii in summary]
            summary = ''.join(summary)
            citations = doc.xpath('//us-references-cited/us-citation/patcit/document-id/doc-number/text()')
            #test1 = doc.xpath('//description/p/text()')          
            #test2 = doc.xpath('//detailed-description/p/text()')
            #test3 = doc.xpath('//summary-of-invention/p/text()')
            test = True
            if(assignee != None):
                #if any(word in assignee.lower() for word in allcomp):      
                if test == True:
                    if abstract != None and len(abstract) > 100:          
                          abstract_list.append(abstract)
                          company_list.append(assignee)
                          title_list.append(title)
                          docID_list.append(docID)
                          date_list.append(date)
                          claim_list.append(summary)
                          citation_list.append(citations)
                          
with open("abstract_16.txt","wb") as pickle_file:
    pickle.dump(abstract_list, pickle_file)
with open("company_16.txt","wb") as pickle_file2:
    pickle.dump(company_list,pickle_file2)
with open("title_16.txt","wb") as pickle_file3:
    pickle.dump(title_list,pickle_file3)
with open("docID_16.txt","wb") as pickle_file4:
    pickle.dump(docID_list,pickle_file4)
with open("date_16.txt","wb") as pickle_file4:
    pickle.dump(date_list,pickle_file4)
with open("claim_16.txt","wb") as pickle_file4:
    pickle.dump(claim_list,pickle_file4)

end_time = time.time()
print("time taken is" + str(end_time-start_time))