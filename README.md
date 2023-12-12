# Quick introduction of text retrieval and classification by compression distance method
## Overview and Intro
This repository contains a brief overview of text retrieval and classification using the "Normalized Compression Distance" (NCD) method introduced by [“Low-Resource” Text Classification: A Parameter-Free Classification Method with Compressors](https://aclanthology.org/2023.findings-acl.426/). The authors created a new text classification method and library of [npc_gzip](https://github.com/bazingagin/npc_gzip) based on the compression distance measure introduced by ["The Similarity Metric"](https://www.researchgate.net/publication/220679130_The_Similarity_Metric). They add normalize feature for the compression distance method, and implemented it into K-Nearest Neighbors classification function. The authors claim that the NCD method offers a competitive alternative to existing classification methods by achieving comparable accuracy with less computational overhead.
The NCD measure does not require feature engineering and hyperparameter tuning, and it also does not need massive GPU power since it is run by gzip compressing and simple sorting algorithm.
This repository contains some test codes and notebook files that applies NCD method into: text retrieval (Part1), and hans-on text classification (Part2).
I found the implementation of this method is comprehensive, but found some challenges.

## Part 1: Text Retrieval
Please refer [Part1 Notebook](part1/part1.ipynb)
### NCD Measure for Query-Document Text Retrieval
The Normalized Compression Distance (NCD) is NOT designed for text retrieval in the paper. 
In this section, I tried to apply this method into simple text retrieval task with simple and small dataset.
The compression_search function that I drafted measures the dissimilarity between a query and a document by comparing their compressed sizes using a compression algorithm like GZip. Lower NCD values indicate higher similarity between the query and the document.
This section is direct reference of the TIS class materials such as overall text retrieval idea and scoring methods used in the class.
I use BM25 algorithm for the baseline and use nDCG for benchmark of Cranfield dataset and MAP and MRR for CISI.

### Cranfield and CISI Dataset
This analysis uses two benchmark datasets for text retrieval: Cranfield and CISI.
- Cranfield: A classic dataset which consists of technical documents and corresponds to queries, widely used for evaluating retrieval performance.
    - The dataset has already been modified for the course and this repository uses this version
- CISI: Another established dataset with a different characteristic, offering a broader perspective on NCD's retrieval capabilities.

### Outcome and Challenges
The performance from NCD method is far from 


## Part 2: Text Classification
Please refer [Part2 Notebook](part2/part2.ipynb)
### Dataset
I used [Hierarchical text classification] (https://www.kaggle.com/datasets/kashnitsky/hierarchical-text-classification) which has a similar characteristics in terms of length and number of words and labels compared to AGNews which is used in the paper.
For the simplification, I only used Cat1 (level1) label.

### npc-gzip library

### KNN, SVM, and BERT

### Outcome and Challenges

## Conclusion

## Contributor
- K Kokubun(UID: kokubun3)

## References
- ["The Similarity Metric"](https://www.researchgate.net/publication/220679130_The_Similarity_Metric)
- [“Low-Resource” Text Classification: A Parameter-Free Classification Method with Compressors](https://aclanthology.org/2023.findings-acl.426/)
- [npc_gzip](https://github.com/bazingagin/npc_gzip)
- [Cranfield collection](http://ir.dcs.gla.ac.uk/resources/test_collections/cran/)
- [CISI Dataset] (https://www.kaggle.com/datasets/dmaso01dsta/cisi-a-dataset-for-information-retrieval)
- [Hierarchical text classification] (https://www.kaggle.com/datasets/kashnitsky/hierarchical-text-classification)
