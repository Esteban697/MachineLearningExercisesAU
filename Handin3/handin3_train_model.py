# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 21:44:43 2018

@author: esteb
"""
###libraries
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pprint import pprint
###Other codes
import handin3_hmm as hmm3
import compare_anns as comp
import text_to_fasta as ttf

def read_genes_train(filename1, filename2, n_samples):
    #calculate the probabilities for the model
    file_path1='data-handin3/' + filename1 + '.fa'
    file_path2='data-handin3/' + filename2 + '.fa'
    sequence = hmm3.read_fasta_file(file_path1)
    annot = hmm3.read_fasta_file(file_path2)
    if n_samples == None:
        gen = sequence[filename1]
        ann = annot[filename2]
    else:
        gen = sequence[filename1][0:(n_samples-1)]
        ann = annot[filename2][0:(n_samples-1)]

    print('\n','Observable States')
    states = ['C', 'A', 'T', 'G']
    pi = [hmm3.calculate_p('C',gen),
          hmm3.calculate_p('A',gen),
          hmm3.calculate_p('T',gen),
          hmm3.calculate_p('G',gen)]
    state_space = pd.Series(pi, index=states, name='states')
    print(state_space)
    
    print('\n','Initial Probabilities')
    hidden_states = ['C', 'N', 'R']
    pi = [hmm3.calculate_p('C',ann),
          hmm3.calculate_p('N',ann),
          hmm3.calculate_p('R',ann)]
    state_space = pd.Series(pi, index=hidden_states, name='states')
    print(state_space)
    
    print('\n','Transition Probabilities')
    a_df = pd.DataFrame(columns=hidden_states, index=hidden_states)
    a_df.loc[hidden_states[0]], a_df.loc[hidden_states[1]], a_df.loc[hidden_states[2]] = hmm3.trans_probs_3_matrix(ann)
    print(a_df)
    trans_probs = a_df.values
    
    print('\n','Emission Probabilities')
    observable_states = states
    b_df = pd.DataFrame(columns=observable_states, index=hidden_states)
    b_df.loc[hidden_states[0]],b_df.loc[hidden_states[1]],b_df.loc[hidden_states[2]] = hmm3.emission_probs_matrix(gen,ann)
    print(b_df)
    emis_probs = b_df.values
    return pi, trans_probs, emis_probs, ann, gen
    
for i in range(5):
    name1='genome' + str(i+1)
    name2='true-ann' + str(i+1)
    print('\n','Probabilities for: ' + name1)
    pi_new, trans_probs_new, emis_probs_new, ann, gen = read_genes_train(name1, name2, None)
    if i==0:
        pi=pi_new
        trans=trans_probs_new
        emission=emis_probs_new
    else:
        for i in range(len(pi)):
            pi[i]=(pi[i]+pi_new[i])/2
        for i in range(len(trans)):
            for j in range(len(trans[0])):
                trans[i][j]=(trans[i][j]+trans_probs_new[i][j])/2
        for i in range(len(emission)):
            for j in range(len(emission[0])):
                emission[i][j]=(emission[i][j]+emis_probs_new[i][j])/2
print('\n','Initial probabilities (Mean)')
print(pi)
print('\n','Transition probabilities(Mean)')
print(trans)
print('\n','Emission probabilities(Mean)')
print(emission)
gen_number=1
filename1='genome' + str(gen_number)
file_path1='data-handin3/' + filename1 + '.fa'
sequence = hmm3.read_fasta_file(file_path1)
gen = sequence[filename1]
ind = hmm3.translate_observations_to_indices(gen)
path, delta, phi = hmm3.viterbi(pi, trans, emission, ind)
path_letters = hmm3.translate_indices_to_guesses(path) #predictions saved
if gen_number >= 6:
    f= open("gen" + str(gen_number) + ".txt","w+")
    f.write(path_letters)
    f.close()
    ttf.convert("gen" + str(gen_number),"true-ann" + str(gen_number))
filename2='true-ann' + str(gen_number)
file_path2=filename2 + '.fasta'
annotations = hmm3.read_fasta_file(file_path2)
comp.print_all(annotations[filename2],path_letters)

