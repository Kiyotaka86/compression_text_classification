import numpy as np
import pandas as pd
import gzip

# Define the function to calculate the NCD and return the indexes of the top k smallest distances
def compressed_classification(query, text, k):
    x1 = query 
    Cx1 = len(gzip.compress(x1.encode())) 
    distance_from_x1 = []
    for (x2) in text.values(): # 
        Cx2 = len(gzip.compress(x2.encode()))
        x1x2 = " ".join([x1, x2])
        Cx1x2 = len(gzip.compress(x1x2.encode()))
        ncd = (Cx1x2 - min(Cx1,Cx2)) / max(Cx1, Cx2) # calculate the normalized compression distance
        distance_from_x1.append(ncd)
    # Directly return the indexes of the top k smallest distances
    return np.argsort(np.array(distance_from_x1))[:k] 

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


# Declare K
k = 10

# Execute the test and store the result
test_result = {}
for key in qry_set.keys():
    if key not in test_result:
        test_result[key] = []
    # passing each query to the classification function with the whole text and k
    test_result[key].append(compressed_classification(qry_set[key], documents, k))

# Score the result by matching the result with the qrels
scored_result = {}
for key in test_result.keys():
    if key in rel_set.keys():
        if key not in scored_result.keys():
            scored_result[key] = []
        for test in test_result[key]:
            for i in test:
                # i+1 because the index starts from 0
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

# MAP = 0.07763157894736843
# MRR = 0.13420530492898913