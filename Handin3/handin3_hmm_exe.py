# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 12:43:16 2018

@author: esteban struve

Project 3 - Machine Learning course Fall 2018
Hidden Markov Models to predict genomes
"""

import handin3_hmm as hmm3
import compare_anns as comp
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pprint import pprint 

def read_genes(filename1, filename2, n_samples):
    #training 
    file_path1='data-handin3/' + filename1 + '.fa'
    file_path2='data-handin3/' + filename2 + '.fa'
    sequence = hmm3.read_fasta_file(file_path1)
    annot = hmm3.read_fasta_file(file_path2)
    gen = sequence[filename1][0:n_samples]
    ann = annot[filename2][0:n_samples]
    ind = hmm3.translate_observations_to_indices(gen)
    #ann = "CCCCCCNNNNNNNNRRRRRR"
    #gen = "CGATTAAAGATAGAAATACA"
    #print('\n','Observable States')
    states = ['C', 'A', 'T', 'G']
    pi = [hmm3.calculate_p('C',gen),
          hmm3.calculate_p('A',gen),
          hmm3.calculate_p('T',gen),
          hmm3.calculate_p('G',gen)]
    state_space = pd.Series(pi, index=states, name='states')
    #print(state_space)
    
    #print('\n','Hidden States')
    hidden_states = ['C', 'N', 'R']
    pi = [hmm3.calculate_p('C',ann),
          hmm3.calculate_p('N',ann),
          hmm3.calculate_p('R',ann)]
    state_space = pd.Series(pi, index=hidden_states, name='states')
    #print(state_space)
    
    #print('\n','Transition Probabilities')
    a_df = pd.DataFrame(columns=hidden_states, index=hidden_states)
    a_df.loc[hidden_states[0]], a_df.loc[hidden_states[1]], a_df.loc[hidden_states[2]] = hmm3.trans_probs_3_matrix(ann)
    #print(a_df)
    trans_probs = a_df.values
    
    #print('\n','Emission Probabilities')
    observable_states = states
    b_df = pd.DataFrame(columns=observable_states, index=hidden_states)
    b_df.loc[hidden_states[0]],b_df.loc[hidden_states[1]],b_df.loc[hidden_states[2]] = hmm3.emission_probs_matrix(gen,ann)
    #print(b_df)
    emis_probs = b_df.values
    
    path, delta, phi = hmm3.viterbi(pi, trans_probs, emis_probs, ind)
    
    path_letters = hmm3.translate_indices_to_guesses(path)
    comp.print_all(ann,path_letters)
    
samples = []
acum=50000
for i in range(20):
    acum+=5000
    samples.append(acum)
    read_genes('genome1', 'true-ann1', acum)

