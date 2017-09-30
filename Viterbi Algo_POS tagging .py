import numpy as np
from collections import Counter

file = open("berp-POS-training.txt", "r")
c1 = Counter()
c2 = Counter()
for row in file:
    words = row.strip().split()
    if words:
        c1 += Counter([words[1]])
        c2 += Counter([words[2]])
word_vocabulary = list(c1)
tag_Vocabulary = list(c2)

# Removing the less frequent words from vocabulary
for word in word_vocabulary:
    if c1[word] is 1:
        word_vocabulary.remove(word)

n = len(tag_Vocabulary)
m = len(word_vocabulary)

# Laplace smoothing on transition probability matrix
transition_Count = np.ones((n,n), dtype=np.float64)
emission_Count = np.zeros((n,m), dtype=np.float64)

prev_tag = ''
cur_tag = ''
iteration = 0
file.close()
file2 = open("berp-POS-training.txt", "r")
for rows in file2:
    words = rows.strip().split()
    if words:
        if words[1] in word_vocabulary and words[2] in tag_Vocabulary:
            emission_Count[tag_Vocabulary.index(words[2])][word_vocabulary.index(words[1])] += 1
        if iteration == 0:
            cur_tag = words[2]
        elif iteration > 0:
            prev_tag = cur_tag
            cur_tag = words[2]
            transition_Count[tag_Vocabulary.index(prev_tag)][tag_Vocabulary.index(cur_tag)] += 1
        iteration += 1

emission_Count_col_sum = emission_Count.sum(axis=0)
transition_Count_row_sum = transition_Count.sum(axis=1)
emission_prob = np.divide(emission_Count, emission_Count_col_sum)
transition_Prob = (transition_Count/transition_Count_row_sum.reshape((-1,1)))+1

# Viterbi implementation
obs = []
f = open("test_data.txt", "r")
for row in f:
    words = row.strip().split()
    if words:
        obs.append(words[1])
# obs[] - list of observations (words)
start_prob = np.full((1,n), 1/n, dtype=float)
trellis = np.ones((n, len(obs)), dtype=np.float64)
back_pointer = np.zeros((n, len(obs)), dtype=int)

for state in tag_Vocabulary:
    trellis[tag_Vocabulary.index(state)][0] = (start_prob[0][tag_Vocabulary.index(state)] * emission_prob[tag_Vocabulary.index(state)][word_vocabulary.index(obs[0])])

for obs_index in range(1, len(obs)):
    for state in tag_Vocabulary:
        max_val = 1.0
        for prev_state in tag_Vocabulary:
            temp_max = trellis[tag_Vocabulary.index(prev_state)][obs_index-1]*transition_Prob[tag_Vocabulary.index(prev_state)][tag_Vocabulary.index(state)]
            if temp_max > max_val:
                back_pointer[tag_Vocabulary.index(state)][obs_index] = tag_Vocabulary.index(prev_state)

        if obs[obs_index] in word_vocabulary:
            trellis[tag_Vocabulary.index(state)][obs_index] = max_val * emission_prob[tag_Vocabulary.index(state)][word_vocabulary.index(obs[obs_index])]
        else:
            trellis[tag_Vocabulary.index(state)][obs_index] = max_val * emission_prob.max()

obs_tags_index = []
i = 0
obs_tags_index.append(np.argmax(trellis[:, len(obs)-1]))
for i in range(len(obs)-1, 0, -1):
    obs_tags_index.append(back_pointer[obs_tags_index[-1]][i])

final = (list(reversed(obs_tags_index)))
i = 0
while i<len(obs):
    print(tag_Vocabulary[final[i]])
    i += 1

infile = open("test_data.txt", "r")
outfile = open("Gayam-Prathyusha-assgn2-test-output.txt", "w")
i = 0
for row in infile:
    words = row.strip().split()
    if words:
        outfile.write(words[0]+"\t"+words[1]+"\t"+tag_Vocabulary[final[i]]+"\n")
        i += 1
    else:
        outfile.write("\n")
infile.close()
outfile.close()