import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')


# Load the stopwords
with open('stopwords.txt', 'r') as f:
    stopwords = f.readlines()
    stopwords = [word.strip() for word in stopwords]

def preprocess(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    # Remove stopwords
    filtered_tokens = [word for word in tokens if word not in stopwords]
    return filtered_tokens


with open('cranfield/cranfield.dat', 'r') as f:
    cranfield_lines = f.readlines()
    cranfield_lines = [line.strip() for line in cranfield_lines]
    idx = 1
    cranfield_lines_idx = []
    for line in cranfield_lines:
        cranfield_lines_idx.append((line,idx))
        idx += 1

cranfield_qrels = {}
with open('cranfield/cranfield-qrels.txt', 'r') as f:
    for line in f:
        numbers = [int(n) for n in line.split()]
        key = numbers[0]
        if key not in cranfield_qrels:
            cranfield_qrels[key] = []
        cranfield_qrels[key].append(numbers[1:])

with open('cranfield/cranfield-queries.txt', 'r') as f:
    cranfield_queries = f.readlines()
    idx = 1
    cranfield_queries_idx = []
    for line in cranfield_queries:
        cranfield_queries_idx.append((line,idx))
        idx += 1

# Tokenize the corpus and remove the stopwords
tokenized_corpus = [preprocess(line) for line, _ in cranfield_lines_idx]

# Train the model
bm25 = BM25Okapi(tokenized_corpus)

# Declare K
k = 10

# Execute the test and store the result
test_result = {}
for i in range(len(cranfield_queries_idx)):
    key = i+1
    if key not in test_result:
        test_result[key] = []
    # tokenize the query and get the top k result
    tokenized_query = preprocess(cranfield_queries_idx[i][0])
    # append the result to the test_result
    test_result[key].append(np.argsort(bm25.get_scores(tokenized_query))[::-1][:k]) # [::-1] to sort in descending order

# Score the result by matching the result with the qrels
scored_result = {}
for key in test_result:
    if key not in scored_result:
        scored_result[key] = []
    # test_result[key][0] is the list of indexes of compressed_classification result
    for i in range(len(test_result[key][0])): 
        for j in range(len(cranfield_qrels[key])):
            if test_result[key][0][i] == cranfield_qrels[key][j][0]:
                scored_result[key].append((i+1, cranfield_qrels[key][j][1]))
                break
        if len(scored_result[key]) < i+1:
            scored_result[key].append((i+1, 0))

# Calculate the NDCG
ndcg_result = []
for key in scored_result:
    # Calculate the DCG
    dcg = 0
    for i in range(len(scored_result[key])):
        if i == 0:
            dcg += scored_result[key][i][1] # first element is not divided by log2
        else:
            dcg += scored_result[key][i][1] / np.log2(i+1) # i+1 because the index starts from 0
    # Calculate the IDCG        
    idcg = 0
    length = len(cranfield_qrels[key]) if len(cranfield_qrels[key]) < k else k # if the length of qrels is less than k, use the length of qrels
    sorted_cranfield_qrels = sorted(cranfield_qrels[key], key=lambda x: x[1], reverse=True) # sort the qrels by the score for IDCG
    for i in range(length): # calculate the IDCG
        if i == 0:
            idcg += sorted_cranfield_qrels[i][1]
        else:
            idcg += sorted_cranfield_qrels[i][1] / np.log2(i+1)
    ndcg_result.append(dcg/idcg) # calculate the NDCG and append it to the list

# Print the average NDCG
print(np.mean(np.array(ndcg_result)))
