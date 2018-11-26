# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 23:11:39 2018

@author: esteb
"""
import handin3_hmm as hmm3
import text_to_fasta as ttf

filename1='genome10'
file_path1='data-handin3/' + filename1 + '.fa'
sequence = hmm3.read_fasta_file(file_path1)
gen = sequence[filename1]#[0:100000]
ind = hmm3.translate_observations_to_indices(gen)


path, delta, phi = hmm3.viterbi(pi, trans, emission, ind)
path_letters = hmm3.translate_indices_to_guesses(path)
#comp.print_all(ann,path_letters)
f= open("gen10.txt","w+")
f.write(path_letters)
f.close()
ttf.convert("gen10","true-ann10")
