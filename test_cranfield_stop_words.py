import numpy as np
import pandas as pd
import gzip


def compressed_classification(query, text, k):
    x1, _ = query
    Cx1 = len(gzip.compress(x1.encode())) 
    distance_from_x1 = []
    for (x2,_) in text:
        Cx2 = len(gzip.compress(x2.encode()))
        x1x2 = " ".join([x1, x2])
        Cx1x2 = len(gzip.compress(x1x2.encode()))
        ncd = (Cx1x2 - min(Cx1,Cx2)) / max(Cx1, Cx2)
        distance_from_x1.append(ncd)
    return np.argsort(np.array(distance_from_x1))[:k]  # +1 because the index starts from 1

with open('stopwords.txt', 'r') as f:
    stopwords = f.readlines()
    stopwords = [word.strip() for word in stopwords]

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

test_result = {}
for i in range(len(cranfield_queries_idx)):
    key = i+1
    if key not in test_result:
        test_result[key] = []
    test_result[key].append(compressed_classification(cranfield_queries_idx[i], cranfield_lines_idx, 10))

scored_result = {}
for key in test_result:
    if key not in scored_result:
        scored_result[key] = []
    for i in range(len(test_result[key][0])):
        for j in range(len(cranfield_qrels[key])):
            if test_result[key][0][i] == cranfield_qrels[key][j][0]:
                scored_result[key].append((i+1, cranfield_qrels[key][j][1]))
                break
        if len(scored_result[key]) < i+1:
            scored_result[key].append((i+1, 0))

ndcg_result = []
for key in scored_result:
    dcg = 0
    for i in range(len(scored_result[key])):
        dcg += scored_result[key][i][1] / np.log2(i+2)
    idcg = 0
    length = len(cranfield_qrels[key]) if len(cranfield_qrels[key]) < 10 else 10
    sorted_cranfield_qrels = sorted(cranfield_qrels[key], key=lambda x: x[1], reverse=True)
    for i in range(length):
        idcg += sorted_cranfield_qrels[i][1] / np.log2(i+2)
    ndcg_result.append(dcg/idcg)
print(np.mean(np.array(ndcg_result)))