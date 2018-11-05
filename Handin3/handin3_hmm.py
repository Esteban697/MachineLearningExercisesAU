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

#x = read_fasta_file('data-handin3/genome1.fa')

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

"""Examples"""
obs_example = 'GTTTCCCAGTGTATATCGAGGGATACTACGTGCATAGTAACATCGGCCAA'
obs_example_trans = translate_observations_to_indices(obs_example)
obs = translate_indices_to_observations(translate_observations_to_indices(obs_example))

path_example = '33333333333321021021021021021021021021021021021021'
indices = translate_path_to_indices(path_example)

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
    assert np.sum(model.emission_probs)==len(model.emission_probs)
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







