#!/usr/bin/env python
# coding: utf-8

# # CSE 5334 Programming Assignment 1 (P1)

# ## Fall 2023

# ## Due: 11:59pm Central Time, Friday, September 29, 2023

# In this assignment, you will implement a toy "search engine" in Python. You code will read a corpus and produce TF-IDF vectors for documents in the corpus. Then, given a query string, you code will return the query answer--the document with the highest cosine similarity score for the query. 
# 
# The instructions on this assignment are written in an .ipynb file. You can use the following commands to install the Jupyter notebook viewer. You can use the following commands to install the Jupyter notebook viewer. "pip" is a command for installing Python packages. You are required to use Python 3.5.1 or more recent versions of Python in this project. 
# 
#     pip install jupyter
# 
#     pip install notebook (You might have to use "sudo" if you are installing them at system level)
# 
# To run the Jupyter notebook viewer, use the following command:
# 
#     jupyter notebook P1.ipynb
# 
# The above command will start a webservice at http://localhost:8888/ and display the instructions in the '.ipynb' file.

# ### Requirements

# * This assignment must be done individually. You must implement the whole assignment by yourself. Academic dishonety will have serious consequences.
# * You can discuss topics related to the assignment with your fellow students. But you are not allowed to discuss/share your solution and code.

# ### Dataset

# We use a corpus of 15 Inaugural addresses of different US presidents. We processed the corpus and provided you a .zip file, which includes 15 .txt files.

# ### Programming Language

# 1. You are required to submit a single .py file of your code.
# 
# 2. You are expected to use several modules in NLTK--a natural language processing toolkit for Python. NLTK doesn't come with Python by default. You need to install it and "import" it in your .py file. NLTK's website (http://www.nltk.org/index.html) provides a lot of useful information, including a book http://www.nltk.org/book/, as well as installation instructions (http://www.nltk.org/install.html).
# 
# 3. In programming assignment 1, other than NLTK, you are not allowed to use any other non-standard Python package. However, you are free to use anything from the the Python Standard Library that comes with Python (https://docs.python.org/3/library/).

# ### Tasks

# You code should accomplish the following tasks:
# 
# (1) <b>Read</b> the 15 .txt files, each of which has the transcript of inaugural addresses by different US presidents. The following code does it. Make sure to replace "corpusroot" by your directory where the files are stored. In the example below, "corpusroot" is a sub-folder named "US_Inaugural_Addresses" in the folder containing the python file of the code. 
# 
# In this assignment we ignore the difference between lower and upper cases. So convert the text to lower case before you do anything else with the text. For a query, also convert it to lower case before you answer the query. 

# In[1]:


import os
corpusroot = './US_Inaugural_Addresses'
for filename in os.listdir(corpusroot):
    if filename.startswith('0') or filename.startswith('1'):
        file = open(os.path.join(corpusroot, filename), "r", encoding='windows-1252')
        doc = file.read()
        file.close() 
        doc = doc.lower()


# (2) <b>Tokenize</b> the content of each file. For this, you need a tokenizer. For example, the following piece of code uses a regular expression tokenizer to return all course numbers in a string. Play with it and edit it. You can change the regular expression and the string to observe different output results. 
# 
# For tokenizing the inaugural Presidential speeches, we will use RegexpTokenizer(r'[a-zA-Z]+')
# 

# In[2]:


from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
tokens = tokenizer.tokenize("CSE4334 and CSE5534 are taught together. IE3013 is an undergraduate course.")
print(tokens)


# (3) Perform <b>stopword removal</b> on the obtained tokens. NLTK already comes with a stopword list, as a corpus in the "NLTK Data" (http://www.nltk.org/nltk_data/). You need to install this corpus. Follow the instructions at http://www.nltk.org/data.html. You can also find the instruction in this book: http://www.nltk.org/book/ch01.html (Section 1.2 Getting Started with NLTK). Basically, use the following statements in Python interpreter. A pop-up window will appear. Click "Corpora" and choose "stopwords" from the list.

# In[3]:


#import nltk
#nltk.download()


# After the stopword list is downloaded, you will find a file "english" in folder nltk_data/corpora/stopwords, where folder nltk_data is the download directory in the step above. The file contains 179 stopwords. nltk.corpus.stopwords will give you this list of stopwords. Try the following piece of code.

# In[4]:


from nltk.corpus import stopwords
print(stopwords.words('english'))
#print(len(stopwords.words('english')))


# (4) Also perform <b>stemming</b> on the obtained tokens. NLTK comes with a Porter stemmer. Try the following code and learn how to use the stemmer.

# In[5]:


from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
print(stemmer.stem('studying'))
print(stemmer.stem('vector'))
print(stemmer.stem('entropy'))
print(stemmer.stem('hispanic'))
print(stemmer.stem('ambassador'))


# (5) Using the tokens, we would like to compute the TF-IDF vector for each document. Given a query string, we can also calculate the query vector and calcuate similarity.
# 
# In the class, we learned that we can use different weightings for queries and documents and the possible choices are shown below:

# <img src = 'weighting_scheme.png'>

# The notation of a weighting scheme is as follows: ddd.qqq, where ddd denotes the combination used for document vector and qqq denotes the combination used for query vector.
# 
# A very standard weighting scheme is: ltc.lnc; where the processing for document and query vectors are as follows:
# Document: logarithmic tf, logarithmic idf, cosine normalization
# Query: logarithmic tf, no idf, cosine normalization
# 
# Implement query-document similarity using the <b>ltc.lnc</b> weighting scheme and show the outputs for the following:

# In[8]:


import os
corpusroot = './US_Inaugural_Addresses'
for filename in os.listdir(corpusroot):
    if filename.startswith('0') or filename.startswith('1'):
        file = open(os.path.join(corpusroot, filename), "r", encoding='windows-1252')
        doc = file.read()
        file.close() 
        doc = doc.lower()


# In[9]:


from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()


# In[10]:


def preprocess_text (doc):
    tokens = tokenizer.tokenize(doc)
    stop_words = stopwords.words('english')
    preprocessed_tokens =[stemmer.stem(token) for token in tokens if token not in stop_words]
    return preprocessed_tokens


# In[11]:


from collections import Counter
import math
presidential_speech = {}
dfs = {}
idfs = {}
total_word_counts = {}
def calculate_idf():
    ndoc = 0
    for filename in os.listdir(corpusroot):
        if filename.endswith('.txt'):
            file_path = os.path.join(corpusroot, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                doc = file.read().lower()
                preprocessed_tokens = preprocess_text(doc)
                tfvec = Counter(preprocessed_tokens)
                presidential_speech[filename] = tfvec
                ndoc += 1
                for token in tfvec:
                    if token not in dfs:
                        dfs[token] = 1
                        total_word_counts[token] = tfvec[token]
                    else:
                        dfs[token] += 1
                        total_word_counts[token] += tfvec[token]
    #print("dfs",dfs)
    for token, df in dfs.items():
        idfs[token] = math.log10(ndoc / df)
    #print("idfs",idfs)
calculate_idf()

def idffunc(stem_token):
    if stem_token not in idfs:
        return -1
    else:
        return idfs[stem_token]


# In[12]:


def getidf(token):
    stem_token = stemmer.stem(token)
    return idffunc(stem_token)


# In[13]:


print(getidf("university"))
print(getidf("city"))
print(getidf("texas"))


# In[14]:


def calculate_tfidf(tfvec):
    sumOfSquares = 0
    tfidf_dic = {}
    for token, tf in tfvec.items():

        idf = idffunc(token)
        tfidf = (1 + math.log10(tf)) * idf
        tfidf_dic[token] = tfidf
        sumOfSquares += math.pow(tfidf,2)
    #print(sumOfSquares)
    norm_factor = math.sqrt(sumOfSquares)

    for token in tfvec:
        tfidf_dic[token]  = tfidf_dic[token]/norm_factor

    return tfidf_dic

def getweight(filename, token):
    stemmed_token = stemmer.stem(token)
    tfvec = presidential_speech[filename]
    tfvec = calculate_tfidf(tfvec)
    #print('normalized tfvec',tfvec)
    if stemmed_token in tfvec:
        return tfvec[stemmed_token]
    else:
        return 0


# In[15]:


print(getweight('02_washington_1793.txt', 'token'))


# In[20]:


import operator
speechvecs = {}

for filename, tfvec in presidential_speech.items():
    tfidfvec = calculate_tfidf(tfvec)
    speechvecs[filename] = tfidfvec


# In[21]:


print(getweight('02_washington_1793.txt', "arrive"))
print(getweight('01_washington_1789.txt', "downtowns"))


# In[22]:


def cosinesim(vec1, vec2):
    common_terms = set(vec1) & set(vec2)
    dot_product = sum(vec1[token] * vec2[token] for token in common_terms)
    magnitude_vec1 = math.sqrt(sum(vec1[token] ** 2 for token in vec1))
    magnitude_vec2 = math.sqrt(sum(vec2[token] ** 2 for token in vec2))

    if magnitude_vec1 > 0 and magnitude_vec2 > 0:
        similarity = dot_product / (magnitude_vec1 * magnitude_vec2)
    else:
        similarity = 0.0

    return similarity

def query(qstring):
    qvec = calculate_tfidf(Counter(preprocess_text(qstring.lower())))
    scores = {filename: cosinesim(qvec, tfidfvec) for filename, tfidfvec in speechvecs.items()}
    result_document = max(scores.items(), key=operator.itemgetter(1))[0]
    result_score = scores[result_document]
    formatted_result = (result_document, result_score)
    return formatted_result


# In[23]:


print(query("City Arlington Texas"))


# In[25]:


print("%.12f" % getidf('british'))
print("%.12f" % getidf('union'))
print("%.12f" % getidf('war'))
print("%.12f" % getidf('military'))
print("%.12f" % getidf('great'))
print("--------------")
print("%.12f" % getweight('02_washington_1793.txt','arrive'))
print("%.12f" % getweight('07_madison_1813.txt','war'))
print("%.12f" % getweight('12_jackson_1833.txt','union'))
print("%.12f" % getweight('09_monroe_1821.txt','british'))
print("%.12f" % getweight('05_jefferson_1805.txt','public'))
print("--------------")
print("(%s, %.12f)" % query("pleasing people"))
print("(%s, %.12f)" % query("british war"))
print("(%s, %.12f)" % query("false public"))
print("(%s, %.12f)" % query("people institutions"))
print("(%s, %.12f)" % query("violated willingly"))
          


# <b> Submit through Canvas your source code in a single .py file.</b> You can use any standard Python library. The only non-standard library/package allowed for this assignment is NLTK. You .py file must define at least the following functions:
# 
# * getidf(token): return the inverse document frequency of a token. If the token doesn't exist in the corpus, return -1. You should stem the parameter 'token' before calculating the idf score.
# 
# * getweight(filename,token): return the normalized TF-IDF weight of a token in the document named 'filename'. If the token doesn't exist in the document, return 0. You should stem the parameter 'token' before calculating the tf-idf score.
# 
# * query(qstring): return a tuple in the form of (filename of the document, score), where the document is the query answer with respect to the weighting scheme. You should stem the query tokens before calculating similarity.

# ### Evaluation

# Your program will be evaluated using the following criteria: 
# 
# * Correctness (75 Points)
# 
# We will evaluate your code by calling the functions specificed above (getidf - 20 points; getweight - 25 points; query - 30 points). So, make sure to use the same function names, parameter names/types/orders as specified above. We will use the above test cases and other queries and tokens to test your program.
# 
# 
# * Preprocessing, Efficiency, modularity (25 Points)
# 
# You should correctly follow the preprocessing steps. An efficient solution should be able to answer a query in a few seconds, you will get deductions if you code takes too long to run (>1 minute). Also, it should consider the boundary cases. Your program should behave correctly under special cases and even incorrect input. Follow good coding standards to make your program easy to understand by others and easy to maintain/extend. 
# 
