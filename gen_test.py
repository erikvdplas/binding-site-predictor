import numpy as np
import random
import time

motifs = [l.strip() for l in open('motifs.txt').readlines()]

L = dict({'A':0,'a':0,'C':1,'c':1,'G':2,'g':2,'T':3,'t':3})

SEQ_LEN = 200

def rand_seq(size=SEQ_LEN):
    values = ['A', 'C', 'G', 'T']
    seq = ''
    for i in range(size):
        seq += random.choice(values)

    while next((motif for motif in motifs if motif in seq), None) is not None:
        match_idx = next(idx for idx, motif in enumerate(motifs) if motif in seq)
        match_len = len(motifs[match_idx])
        match_loc = seq.find(motifs[match_idx])
        seq = seq[:match_loc] + rand_seq(match_len) + seq[match_loc + match_len:]

    return seq

def encode(seq):
    new = np.zeros((1, len(seq), 4))
    for idx, c in enumerate(seq):
        new[:, idx, L[c]] = 1
    return new

for motif in motifs:
    leading = rand_seq()
    trailing = rand_seq()

    first_len = random.randint(0, SEQ_LEN - len(motif))
    center_start = rand_seq(first_len)
    final_len = SEQ_LEN - (len(motif) + first_len)
    center_end = rand_seq(final_len)
    center = center_start + motif + center_end

    final = encode(leading + center + trailing)

    #TODO: Export to pickle and torch whatevah

for idx in range(999 * len(motifs)):
    leading = rand_seq()
    trailing = rand_seq()
    center = rand_seq()

    final = encode(leading + center + trailing)
