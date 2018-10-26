# Torch imports
import torch
from torch.utils.data import Dataset, DataLoader

# Utility imports
import joblib
import os
import math
import numpy as np

DREAM_DATA_PATH = os.environ['DREAM_DATA'] if 'DREAM_DATA' in os.environ else ''

def array(path):
    return joblib.load(path)

def save_array(array, path):
    joblib.dump(array, path, protocol=2)

def dream_array(dream_path):
    return array(DREAM_DATA_PATH + dream_path)

def load_batches(chip_path, batch_size, number=None):
    peaks = array(chip_path)
    dna = dream_array('dna.pkl')

    if number is not None:
        peaks = peaks[:(batch_size * number)]
        dna = dna[:(batch_size * number)]

    batches = []
    for batch_idx in range(math.ceil(len(peaks) / batch_size)):
        peak_seq = []
        dna_seq = []

        for i in range(batch_size):
            peak_array = np.array([peaks[batch_idx * batch_size + i]]).astype('float')
            peak_seq.append(torch.from_numpy(peak_array))
            dna_array = np.unpackbits(np.array(dna[batch_idx * batch_size + i]), -1)
            dna_seq.append(torch.from_numpy(dna_array))

        peak_batch = torch.cat(peak_seq)
        dna_batch = torch.cat(dna_seq)

        peak_batch = peak_batch.view((1, batch_size, 1)).float()        # (seq_len, batch, input_size)
        dna_batch = dna_batch.transpose(0, 2).transpose(1, 2).float()   # (seq_len, batch, input_size)

        batches.append((dna_batch, peak_batch))

    return batches