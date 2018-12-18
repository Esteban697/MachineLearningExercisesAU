# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 13:31:03 2018

@author: esteban struve

Project 3 - Machine Learning course Fall 2018
Hidden Markov Models to predict genomes
"""
import numpy as np

def read_fasta_file(filename):
    """
    Reads the given FASTA file f and returns a dictionary of sequences.

    Lines starting with ';' in the FASTA file are ignored.
    """
    sequences_lines = {}
    current_sequence_lines = None
    with open(filename) as fp:
        for line in fp:
            line = line.strip()
            if line.startswith(';') or not line:
                continue
            if line.startswith('>'):
                sequence_name = line.lstrip('>')
                current_sequence_lines = []
                sequences_lines[sequence_name] = current_sequence_lines
            else:
                if current_sequence_lines is not None:
                    current_sequence_lines.append(line)
    sequences = {}
    for name, lines in sequences_lines.items():
        sequences[name] = ''.join(lines)
    return sequences

"""Example model structure in week 10 exercises"""

def translate_observations_to_indices(obs):
    mapping = {'a': 0, 'c': 1, 'g': 2, 't': 3}
    return [mapping[symbol.lower()] for symbol in obs]

def translate_indices_to_observations(indices):
    mapping = ['a', 'c', 'g', 't']
    return ''.join(mapping[idx] for idx in indices)

def translate_path_to_indices(path):
    return list(map(lambda x: int(x), path))

def translate_indices_to_path(indices):
    return ''.join([str(i) for i in indices])

def translate_indices_to_guesses(indices):
    mapping = ['c', 'n', 'r']
    return ''.join(mapping[int(idx)] for idx in indices)

"""Examples"""
obs_example = 'GTTTCCCAGTGTATATCGAGGGATACTACGTGCATAGTAACATCGGCCAA'
obs_example_trans = translate_observations_to_indices(obs_example)
obs = translate_indices_to_observations(translate_observations_to_indices(obs_example))

path_example = '33333333333321021021021021021021021021021021021021'
indices = translate_path_to_indices(path_example)

gen1 = read_fasta_file('data-handin3/genome1.fa')

"""HMM"""
class hmm:
    def __init__(self, init_probs, trans_probs, emission_probs):
        self.init_probs = init_probs
        self.trans_probs = trans_probs
        self.emission_probs = emission_probs

"""Example HMM 1"""
init_probs_7_state = [0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00]

trans_probs_7_state = [
    [0.00, 0.00, 0.90, 0.10, 0.00, 0.00, 0.00],
    [1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.05, 0.90, 0.05, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00],
    [0.00, 0.00, 0.00, 0.10, 0.90, 0.00, 0.00],
]

emission_probs_7_state = [
    #   A     C     G     T
    [0.30, 0.25, 0.25, 0.20],
    [0.20, 0.35, 0.15, 0.30],
    [0.40, 0.15, 0.20, 0.25],
    [0.25, 0.25, 0.25, 0.25],
    [0.20, 0.40, 0.30, 0.10],
    [0.30, 0.20, 0.30, 0.20],
    [0.15, 0.30, 0.20, 0.35],
]

# Collect the matrices in a class.
hmm_7_state = hmm(init_probs_7_state, trans_probs_7_state, emission_probs_7_state)

# We can now reach the different matrices by their names. E.g.:
hmm_7_state.trans_probs

"""Example HMM 2"""
init_probs_3_state = [0.10, 0.80, 0.10]

trans_probs_3_state = [
    [0.90, 0.10, 0.00],
    [0.05, 0.90, 0.05],
    [0.00, 0.10, 0.90],
]

emission_probs_3_state = [
    #   A     C     G     T
    [0.40, 0.15, 0.20, 0.25],
    [0.25, 0.25, 0.25, 0.25],
    [0.20, 0.40, 0.30, 0.10],
]

hmm_3_state = hmm(init_probs_3_state, trans_probs_3_state, emission_probs_3_state)

"""Validation function for probabilities in HMM"""

def validate_hmm(model):
    assert np.sum(model.init_probs)==1.0
    assert np.sum(model.trans_probs)==len(model.trans_probs)
    #assert np.sum(model.emission_probs)==len(model.emission_probs)
    if True in ((np.array(model.init_probs))<0):
        print("Found value is below range in initial probabilities")
        return
    if True in ((np.array(model.init_probs))>1):
        print("Found value is above range in initial probabilities")
        return
    if True in ((np.array(model.trans_probs))<0):
        print("Found value is below range in transition probabilities")
        return
    if True in ((np.array(model.trans_probs))>1):
        print("Found value is above range in transition probabilities")
        return
    if True in ((np.array(model.emission_probs))<0):
        print("Found value is below range in emission probabilities")
        return
    if True in ((np.array(model.emission_probs))>1):
        print("Found value is above range in emission probabilities")
        return
    print("Valid probailities for this model")
    
def calculate_p(value,input_string):
    #count=0
    result=0
    #for c in input_string:
    #    if c==value:
    #        count+=1
    #result = count/len(input_string)
    if input_string[0]==value:
        result=1
    return result
    
def calculate_pxz(x_value,z_value,string_input_x,string_input_z):
    assert len(string_input_x) == len(string_input_z)
    count1=0
    count2=0
    index=0
    for c in string_input_x:
        if string_input_z[index]==z_value:
            count2+=1
            if c==x_value:
                count1+=1
        index+=1
    if count2==0:
        print("The Z value {} is not found in the string".format(z_value))
        return -1
    prob_xgivenz=count1/count2
    return prob_xgivenz

def calculate_pztoz(zi_prev,zi,string_input_z):
    count1=0
    count2=0
    for i in range(len(string_input_z)-1):
        if string_input_z[i]==zi_prev:
            count2+=1
            if string_input_z[i+1]==zi:
                count1+=1
    if count2==0:
        print("The Z value {} is not found in the string".format(string_input_z))
        return -1
    return count1/count2

def trans_probs_3_matrix(string_input_z):
    row1 = [calculate_pztoz("C","C",string_input_z),
               calculate_pztoz("C","N",string_input_z),
               calculate_pztoz("C","R",string_input_z)]
    row2 = [calculate_pztoz("N","C",string_input_z),
               calculate_pztoz("N","N",string_input_z),
               calculate_pztoz("N","R",string_input_z)]
    row3 = [calculate_pztoz("R","C",string_input_z),
               calculate_pztoz("R","N",string_input_z),
               calculate_pztoz("R","R",string_input_z)]
    return row1,row2,row3

def emission_probs_matrix(string_input_x,string_input_z):
    row1 = [calculate_pxz("A","C",string_input_x,string_input_z),
               calculate_pxz("C","C",string_input_x,string_input_z),
               calculate_pxz("T","C",string_input_x,string_input_z),
               calculate_pxz("G","C",string_input_x,string_input_z)]
    row2 = [calculate_pxz("A","N",string_input_x,string_input_z),
               calculate_pxz("C","N",string_input_x,string_input_z),
               calculate_pxz("T","N",string_input_x,string_input_z),
               calculate_pxz("G","N",string_input_x,string_input_z)]
    row3 = [calculate_pxz("A","R",string_input_x,string_input_z),
               calculate_pxz("C","R",string_input_x,string_input_z),
               calculate_pxz("T","R",string_input_x,string_input_z),
               calculate_pxz("G","R",string_input_x,string_input_z)]
    return row1,row2,row3

def viterbi(pi, a, b, obs):
    
    nStates = np.shape(b)[0]
    T = np.shape(obs)[0]
    
    # init blank path
    path = np.zeros(T)
    # delta --> highest probability of any path that reaches state i
    delta = np.float64(np.zeros((nStates, T)))
    # phi --> argmax by time step for each state
    phi = np.zeros((nStates, T))
    
    # init delta and phi 
    delta[:, 0] = pi * b[:, obs[0]]
    phi[:, 0] = 0

    print('\nStart Walk Forward\n')    
    # the forward algorithm extension
    for t in range(1, T):
        for s in range(nStates):
            #delta[s, t] = np.max(delta[:, t-1] * np.log(a[:, s])) * np.log(b[s, obs[t]])
            delta[s, t] = np.max(delta[:, t-1] * a[:, s]) * b[s, obs[t]] 
            phi[s, t] = np.argmax(delta[:, t-1] * a[:, s])
            #print('s={s} and t={t}: phi[{s}, {t}] = {phi}'.format(s=s, t=t, phi=phi[s, t]))
    
    # find optimal path
    print('-'*50)
    print('Start Backtrace\n')
    path[T-1] = np.argmax(delta[:, T-1])
    #p('init path\n    t={} path[{}-1]={}\n'.format(T-1, T, path[T-1]))
    for t in range(T-2, -1, -1):
        path[t] = phi[int(path[t+1]), int(t+1)]
        #p(' '*4 + 't={t}, path[{t}+1]={path}, [{t}+1]={i}'.format(t=t, path=path[t+1], i=[t+1]))
        #print('path[{}] = {}'.format(t, path[t]))
    return path, delta, phi

def viterbi_log(pi, a, b, obs):
    
    nStates = np.shape(b)[0]
    T = np.shape(obs)[0]
    a = np.float64(a)
    b = np.float64(b)
    # init blank path
    path = np.zeros(T)
    # delta --> highest probability of any path that reaches state i
    delta = np.float64(np.zeros((nStates, T)))
    # phi --> argmax by time step for each state
    phi = np.float64(np.zeros((nStates, T)))
    
    # init delta and phi 
    delta[:, 0] = pi * b[:, obs[0]]
    phi[:, 0] = 0

    print('\nStart Walk Forward\n')    
    # the forward algorithm extension
    for t in range(1, T):
        for s in range(nStates):
            #delta[s, t] = np.max(delta[:, t-1] * np.log(a[:, s])) * np.log(b[s, obs[t]])
            delta[s, t] = np.max(np.log(delta[:, t-1] * a[:, s])) * np.log(b[s, obs[t]]) 
            phi[s, t] = np.argmax(np.log(delta[:, t-1] * a[:, s]))
            #print('s={s} and t={t}: phi[{s}, {t}] = {phi}'.format(s=s, t=t, phi=phi[s, t]))
    
    # find optimal path
    print('-'*50)
    print('Start Backtrace\n')
    path[T-1] = np.argmax(delta[:, T-1])
    #p('init path\n    t={} path[{}-1]={}\n'.format(T-1, T, path[T-1]))
    for t in range(T-2, -1, -1):
        path[t] = phi[int(path[t+1]), int(t+1)]
        #p(' '*4 + 't={t}, path[{t}+1]={path}, [{t}+1]={i}'.format(t=t, path=path[t+1], i=[t+1]))
        #print('path[{}] = {}'.format(t, path[t]))
    return path, delta, phi






