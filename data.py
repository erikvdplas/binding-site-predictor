# Torch imports
import torch
from torch.utils.data import Dataset, DataLoader

# Utility imports
import joblib
import os
import math
import numpy as np
import random

DREAM_DATA_PATH = os.environ['DREAM_DATA'] if 'DREAM_DATA' in os.environ else ''

def array(path):
    return joblib.load(path)

def save_array(array, path):
    joblib.dump(array, path, protocol=2)

def dream_array(dream_path):
    return array(DREAM_DATA_PATH + dream_path)

class BSDataset:
    SEQ_STRIDE = 50
    SEQ_LEN = 200

    def __init__(self, chip_path, batch_size, test_part=0.2, adjacent_length=16):
        self.adjacent_length = adjacent_length

        self.peaks_raw = array(chip_path)
        p_len = len(self.peaks_raw)
        self.dna_raw = dream_array('dna.pkl')
        self.batch_size = batch_size
        self.train_offset = 0

        total_chunks = int(p_len / self.adjacent_length)

        self.test_indices = [[i * self.adjacent_length + j for j in range(self.adjacent_length)]
                             for i in random.sample(range(total_chunks), int(total_chunks * test_part))]
        self.test_indices = [item for sublist in self.test_indices for item in sublist]     # Flatten
        self.validate_test_indices()

        self.test_offset = 0

    def validate_test_indices(self):
        alternatives = []
        for index in self.test_indices[:(16*8*512)]:
            if self.peaks_raw[index] != 0:
                alternatives.append(self.peaks_raw[index])
        print(len(alternatives))

    def load_batches(self, number=1, test=False):
        peaks = []
        dna = []

        for i in range(number * self.batch_size * self.adjacent_length):
            if test:
                if self.test_offset >= len(self.peaks_raw):
                    self.test_offset = 0
                while self.test_offset not in self.test_indices:
                    self.test_offset += 1
                peaks.append(self.peaks_raw[self.test_offset])
                print(self.test_indices)
                dna.append(np.unpackbits(np.array(self.dna_raw[self.test_offset]), -1))
                self.test_offset += 1
            else:
                if self.train_offset >= len(self.peaks_raw):
                    self.train_offset = 0
                while self.train_offset in self.test_indices:
                    self.train_offset += 1
                peaks.append(self.peaks_raw[self.train_offset])
                print(self.train_offset)
                dna.append(np.unpackbits(np.array(self.dna_raw[self.train_offset]), -1))
                self.train_offset += 1

        batches = []
        for ni in range(number):
            for ai in range(self.adjacent_length):
                peak_seq = []
                l_dna_seq = []
                t_dna_seq = []

                for bi in range(self.batch_size):
                    batch_offset = self.adjacent_length * self.batch_size * ni
                    final_offset = batch_offset + self.adjacent_length * bi + ai

                    peak_arr = np.array([peaks[final_offset]]).astype('float')
                    peak_seq.append(torch.from_numpy(peak_arr))

                    dna_arr_l = dna[final_offset]
                    dna_arr_t = dna_arr_l

                    MAX_LEADS = int(self.SEQ_LEN / self.SEQ_STRIDE)

                    # Gather leading items
                    for li in range(min(MAX_LEADS, ai)):
                        offset = final_offset - (li + 1)
                        prefix = dna[offset][:, :, :self.SEQ_STRIDE]
                        dna_arr_l = np.concatenate((prefix, dna_arr_l), axis=2)

                    # Gather trailing items
                    for ti in range(min(MAX_LEADS, self.adjacent_length - (ai + 1))):
                        offset = final_offset + (ti + 1)
                        suffix = dna[offset][:, :, -self.SEQ_STRIDE:]
                        dna_arr_t = np.concatenate((dna_arr_t, suffix), axis=2)

                    l_dna_seq.append(torch.from_numpy(dna_arr_l))
                    t_dna_seq.append(torch.from_numpy(dna_arr_t))

                peak_batch = torch.cat(peak_seq)
                l_dna_batch = torch.cat(l_dna_seq)
                t_dna_batch = torch.cat(t_dna_seq)

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                peak_batch = peak_batch.view((1, self.batch_size, 1)).float().to(device)       # (seq_len, batch, input_size)
                l_dna_batch = l_dna_batch.transpose(0, 2).transpose(1, 2).float().to(device)   # (seq_len, batch, input_size)
                t_dna_batch = t_dna_batch.transpose(0, 2).transpose(1, 2).float().to(device)   # (seq_len, batch, input_size)

                batches.append((l_dna_batch, t_dna_batch, peak_batch))

        return batches