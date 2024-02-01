import nltk
import math
from nltk.stem import PorterStemmer
from collections import defaultdict
from operator import itemgetter

# Sample corpus and document-term matrix
corpus = {
    'doc1': 'This is a sample document for testing.',
    'doc2': 'Another example document for testing.',
    'doc3': 'A third document is here for testing purposes.',
}

# Tokenize and stem the corpus
stemmer = PorterStemmer()
corpus_tokens = {doc_name: [stemmer.stem(token) for token in nltk.word_tokenize(doc.lower())] for doc_name, doc in corpus.items()}

# Compute IDF values
total_documents = len(corpus_tokens)
idf_values = defaultdict(int)

for doc_tokens in corpus_tokens.values():
    unique_tokens = set(doc_tokens)
    for token in unique_tokens:
        idf_values[token] += 1

for token, count in idf_values.items():
    idf_values[token] = math.log(total_documents / (1 + count))

# Function to calculate TF-IDF weight for a token in a document
def get_weight(filename, token):
    if filename in corpus_tokens and token in corpus_tokens[filename]:
        tf = corpus_tokens[filename].count(token)
        return (1 + math.log(tf)) * idf_values.get(token, 0)
    return 0

# Function to calculate inverse document frequency (IDF) of a token
def get_idf(token):
    stemmed_token = stemmer.stem(token)
    if stemmed_token in idf_values:
        return idf_values[stemmed_token]
    return -1

# Function to perform a query
def query(qstring):
    query_tokens = [stemmer.stem(token) for token in nltk.word_tokenize(qstring.lower())]
    results = []

    for doc_name, doc_tokens in corpus_tokens.items():
        score = sum(get_weight(doc_name, token) for token in query_tokens)
        results.append((doc_name, score))

    results.sort(key=itemgetter(1), reverse=True)
    return results

# Example usage
query_string = "sample document"
results = query(query_string)

if results:
    print("Query results for '{}':".format(query_string))
    for doc_name, score in results:
        print("Document: {}, Score: {:.4f}".format(doc_name, score))
else:
    print("No matching documents found for query: '{}'".format(query_string))
