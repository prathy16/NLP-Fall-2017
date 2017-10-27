#Designed Naive Bayes classifier for sentiment analysis

import numpy as np
import pandas as pd
from collections import Counter

df1 = pd.read_csv('hotelNegT-train.txt', delimiter='\t', names=['text'], usecols=[1])
df2 = pd.read_csv('hotelPosT-train.txt', delimiter='\t', names=['text'], usecols=[1])

(NegDocs, col) = df1.shape  # Negative documents
(PosDocs, col) = df2.shape  # Positive documents

# Data Pre Processing
df1["text"] = df1["text"].str.replace("[^\w\s]","")
df2["text"] = df2["text"].str.replace("[^\w\s]","")

dictNegWords = Counter(" ".join(df1["text"].values.tolist()).split(" ")).items()
dictPosWords = Counter(" ".join(df2["text"].values.tolist()).split(" ")).items()

temp_dictNegWords = {}
temp_dictPosWords = {}

# All words are convereted to lower case
for key, value in dictNegWords:
    if key.lower() not in temp_dictNegWords:
        temp_dictNegWords[key.lower()] = value
    else:
        temp_dictNegWords[key.lower()] += value

for key, value in dictPosWords:
    if key.lower() not in temp_dictPosWords:
        temp_dictPosWords[key.lower()] = value
    else:
        temp_dictPosWords[key.lower()] += value

# Vocabulary list of training data
vocabulary = []
for key in temp_dictNegWords:
    if key not in vocabulary:
        vocabulary.append(key)
for key in temp_dictPosWords:
    if key not in vocabulary:
        vocabulary.append(key)

# Laplace smoothing
table = np.ones((len(vocabulary), 2))

for key, value in temp_dictNegWords.items():
    table[vocabulary.index(key)][0] = value
for key, value in temp_dictPosWords.items():
    table[vocabulary.index(key)][1] = value

row_sum_table = table.sum(axis=0)
prob_table = np.divide(table, row_sum_table)

# Prior probabilites of +, - class
Prior_Neg = NegDocs/(NegDocs+PosDocs)
Prior_Pos = PosDocs/(NegDocs+PosDocs)

# Test data
file = open('test_data.txt', 'r')
list_class = []
list_ID = []

for row in file:
    sentence = row.strip().split()
    list_ID.append(sentence[0])
    list_words = sentence[1:]

    # Pre processing of test data by removing punctuations and converting words into lower case
    new_list_words = []
    for word in list_words:
        if word[-1] == ',' or word[-1] == '.' or word[-1] == '!' or word[-1] == '?':
            new_list_words.append(word.replace(word[-1], ''))
        else:
            new_list_words.append(word)

    words = []
    for word in new_list_words:
        if word not in words:
            words.append(word)

    # Probability of class -, + of a sentecne by ignoring stop words
    Neg_Prob = Prior_Neg
    Pos_Prob = Prior_Pos

    for word in words:
        if word in vocabulary and table[vocabulary.index(word)][0] < 250:
            Neg_Prob = Neg_Prob * prob_table[vocabulary.index(word)][0]
        if word in vocabulary and table[vocabulary.index(word)][1] < 250:
            Pos_Prob = Pos_Prob * prob_table[vocabulary.index(word)][1]

    if Neg_Prob > Pos_Prob:
        list_class.append('NEG')
    else:
        list_class.append('POS')

# Writing to output file
Output_file = open("Gayam-Prathyusha-outputfile.txt", "w")
i = 0
while i < len(list_ID):
    Output_file.write(list_ID[i]+"\t"+list_class[i]+"\n")
    i += 1









