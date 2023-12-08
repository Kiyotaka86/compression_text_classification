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


def read_documents ():
    f = open ("CISI/CISI.ALL")
    merged = " "
    # the string variable merged keeps the result of merging the field identifier with its content
    
    for a_line in f.readlines ():
        if a_line.startswith ("."):
            merged += "\n" + a_line.strip ()
        else:
            merged += " " + a_line.strip ()
    # updates the merged variable using a for-loop
    
    documents = {}
    
    content = ""
    doc_id = ""
    # each entry in the dictioanry contains key = doc_id and value = content
    
    for a_line in merged.split ("\n"):
        if a_line.startswith (".I"):
            doc_id = a_line.split (" ") [1].strip()
        elif a_line.startswith (".X"):
            documents[doc_id] = content
            content = ""
            doc_id = ""
        else:
            content += a_line.strip ()[3:] + " "
    f.close ()
    return documents


documents = read_documents ()

with open('CISI/CISI.QRY') as f:
    lines = ""
    for l in f.readlines():
        lines += "\n" + l.strip() if l.startswith(".") else " " + l.strip()
    lines = lines.lstrip("\n").split("\n")
    
qry_set = {}
qry_id = ""
for l in lines:
    if l.startswith(".I"):
        qry_id = l.split(" ")[1].strip()
    elif l.startswith(".W"):
        qry_set[qry_id] = l.strip()[3:]
        qry_id = ""

rel_set = {}
with open('CISI/CISI.REL') as f:
    for l in f.readlines():
        qry_id = l.lstrip(" ").strip("\n").split("\t")[0].split(" ")[0]
        doc_id = l.lstrip(" ").strip("\n").split("\t")[0].split(" ")[-1]
        if qry_id in rel_set:
            rel_set[qry_id].append(doc_id)
        else:
            rel_set[qry_id] = []
            rel_set[qry_id].append(doc_id)


# Tokenize the corpus and remove the stopwords
tokenized_corpus = [preprocess(line) for line in documents.values()]

# Train the model
bm25 = BM25Okapi(tokenized_corpus)

# Declare K
k = 10

# Execute the test and store the result
test_result = {}
for key in qry_set.keys():
    if key not in test_result:
        test_result[key] = []
    # tokenize the query and get the top k result
    tokenized_query = preprocess(qry_set[key])
    # append the result to the test_result
    test_result[key].append(np.argsort(bm25.get_scores(tokenized_query))[::-1][:k]) # [::-1] to sort in descending order


# Score the result by matching the result with the qrels
scored_result = {}
for key in test_result.keys():
    if key in rel_set.keys():
        if key not in scored_result.keys():
            scored_result[key] = []
        for test in test_result[key]:
            for i in test:
                temp = str(i+1)
                if temp in rel_set[key]:
                    scored_result[key].append((i+1, 1))
                else:
                    scored_result[key].append((i+1, 0))
                
# Calculate mean average precision
map_result = []
for key in scored_result.keys():
    temp = 0
    for i in range(len(scored_result[key])):
        if scored_result[key][i][1] == 1:
            temp += 1
    map_result.append(temp/(len(scored_result[key])))
print(np.mean(np.array(map_result)))

# Calculate the MRR
mrr_result = []
for key in scored_result.keys():
    temp = 0
    for i in range(len(scored_result[key])):
        if scored_result[key][i][1] == 1:
            temp = 1/(i+1)
    mrr_result.append(temp)
print(np.mean(np.array(mrr_result)))

# MAP = 0.41228070175438597
# MRR = 0.3530701754385965