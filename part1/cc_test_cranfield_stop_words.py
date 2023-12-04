import numpy as np
import pandas as pd
import gzip
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

# Define the function to calculate the NCD and return the indexes of the top k smallest distances
def compressed_classification(query, text, k):
    x1, _ = query # x1 is the query from cranfield-queries.txt
    Cx1 = len(gzip.compress(x1.encode())) 
    distance_from_x1 = []
    for (x2,_) in text: # x2 is the text of cranfield.dat
        Cx2 = len(gzip.compress(x2.encode()))
        x1x2 = " ".join([x1, x2])
        Cx1x2 = len(gzip.compress(x1x2.encode()))
        ncd = (Cx1x2 - min(Cx1,Cx2)) / max(Cx1, Cx2) # calculate the normalized compression distance
        distance_from_x1.append(ncd)
    # Directly return the indexes of the top k smallest distances
    return np.argsort(np.array(distance_from_x1))[:k] 
# I found indexes distributed cranfield-qrels.txt are already subtracted by 1 from the original cranqrel file, so I don't need to add 1 to the returned index

# Load the stopwords
with open('stopwords.txt', 'r') as f:
    stopwords = f.readlines()
    stopwords = [word.strip() for word in stopwords]

def preprocess(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    # Remove stopwords
    filtered_tokens = [word for word in tokens if word not in stopwords]
    return " ".join(filtered_tokens)

with open('cranfield/cranfield.dat', 'r') as f:
    cranfield_lines = f.readlines()
    cranfield_lines = [line.strip() for line in cranfield_lines]

    cleaned_cranfield_lines = [preprocess(line) for line in cranfield_lines]

    idx = 1
    cranfield_lines_idx = []
    for line in cleaned_cranfield_lines:
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
    cranfield_queries = [line.strip() for line in cranfield_queries]

    cleaned_cranfield_queries = [preprocess(line) for line in cranfield_queries]

    idx = 1
    cranfield_queries_idx = []
    for line in cleaned_cranfield_queries:
        cranfield_queries_idx.append((line,idx))
        idx += 1

# Declare K
k = 10

# Execute the test and store the result
test_result = {}
for i in range(len(cranfield_queries_idx)):
    key = i+1
    if key not in test_result:
        test_result[key] = []
    # passing each query to the classification function with the whole text and k
    test_result[key].append(compressed_classification(cranfield_queries_idx[i], cranfield_lines_idx, k))

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
# nDCG@10 = 0.03016878117012516