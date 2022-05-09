###Purpose
This project aims to provide a POC system for checking claim statements against a set of facts.

###Directory Structure
data # directory for training and testing data
- corpus.txt #training corpus
- fact_base.csv #part of final processed fact database in triples, faster but fewer facts
- fact_base_full.csv # full final processed fact database in triples
- test_set.csv # processed facts for validation test
training_1 #checkpoint to restore for training and inference

Facts Definitions and Information Extraction.ipynb # notebook for processing of corpus to fact_base

Semantic Model Training.ipynb # notebook for model training

main.py # run to make inference

###How to Use
The main.py has two modes. 

Usage 1: "python main.py" for comparing two statements and check if they are contradicting
This mode is faster as it checks only 2 statements.

Usage 2: "python main.py -m kb" for checking against the knowledge base in fact_base.csv
This mode is slower as it checks against many statements in the fact_base