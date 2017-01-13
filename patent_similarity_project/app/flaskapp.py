import gensim
import re
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
import csv


from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/')
def my_form():
   return render_template('patdata.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      title_fl = request.form["title"]
      abstract_fl = request.form["abs"]
      claim_fl = request.form["claim"]            
      return render_template("d3simpat.html")
if __name__ == '__main__':
   app.run(debug = True)