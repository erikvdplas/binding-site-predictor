# Torch imports
import torch

# Utility imports
import os
import numpy as np
import random
import joblib
import time

DREAM_DATA_PATH = os.environ['DREAM_DATA'] if 'DREAM_DATA' in os.environ else ''

def array(path):
    return joblib.load(path)

def save_array(array, path):
    joblib.dump(array, path, protocol=2)

def dream_array(dream_path):
    return array(DREAM_DATA_PATH + dream_path)

def unpack_dna():
    dna_raw = dream_array('dna.pkl')
    unpacked = np.unpackbits(dna_raw, -1)
    joblib.dump(unpacked, DREAM_DATA_PATH + 'dna_unpacked.pkl', protocol=2)

class BSDataset:

    SEQ_STRIDE = 50
    SEQ_LEN = 200

    def __init__(self, chip_path, batch_size, test_part=0.2, adjacent_length=16):
        self.adjacent_length = adjacent_length

        self.peaks_raw = array(chip_path)
        self.peaks_len = len(self.peaks_raw)
        self.dna_raw = dream_array('dna_unpacked.pkl')
        self.batch_size = batch_size
        self.train_offset = 0

        total_chunks = int(self.peaks_len / self.adjacent_length)

        self.test_indices = sorted([i * adjacent_length for i in random.sample(range(total_chunks), int(total_chunks * test_part))])
        # print(self.test_indices[:1000])
        # self.validate_test_indices()

        self.test_count = len(self.test_indices)
        self.test_arr_offset = 0
        self.train_arr_offset = 0

        self.test_offset = 0

    def validate_test_indices(self):
        alternatives = []
        for index in self.test_indices[:(16*16*512)]:
            if self.peaks_raw[index] != 0:
                alternatives.append(self.peaks_raw[index])
        print(len(alternatives))

    def load_batches(self, number=1, test=False):
        peaks = []
        dna = []

        # print(number * self.batch_size * self.adjacent_length)

        # start = time.clock()

        while len(peaks) < number * self.batch_size * self.adjacent_length:
            if test:
                if self.test_arr_offset == self.test_count:
                    self.test_arr_offset = 0
                    self.test_offset = 0

                while self.test_indices[self.test_arr_offset] != self.test_offset:
                    self.test_offset += self.adjacent_length
                    if self.test_offset >= self.peaks_len:
                        self.test_offset = 0

                start_index = self.test_offset
                end_index = self.test_offset + self.adjacent_length

                while self.test_indices[self.test_arr_offset + 1] == end_index:
                    end_index += self.adjacent_length
                    self.test_arr_offset += 1

                self.test_offset = end_index + self.adjacent_length     # Immediately skipping non-test fragment

                # print('Picking %d - %d' % (start_index, end_index - 1))
                peaks += list(self.peaks_raw[start_index:end_index])
                dna += list(self.dna_raw[start_index:end_index])

                self.test_arr_offset += 1
            else:
                while self.test_indices[self.train_arr_offset] == self.train_offset:
                    self.train_offset += self.adjacent_length
                    self.train_arr_offset += 1
                    if self.train_arr_offset == self.test_count:
                        self.train_arr_offset = 0

                start_index = self.train_offset
                end_index = self.train_offset + self.adjacent_length

                while self.test_indices[self.train_arr_offset] != end_index and end_index < self.peaks_len:
                    end_index += self.adjacent_length

                if end_index <= self.peaks_len:
                    self.train_offset = end_index

                    # print('Picking %d - %d' % (start_index, end_index - 1))
                    peaks += list(self.peaks_raw[start_index:end_index])
                    dna += list(self.dna_raw[start_index:end_index])
                else:
                    self.train_offset = 0
                    self.train_arr_offset = 0

        peaks = peaks[:number * self.batch_size * self.adjacent_length]
        dna = dna[:number * self.batch_size * self.adjacent_length]

        # print(time.clock() - start)
        # start = time.clock()

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

                    MAX_LEADS = min(int(self.SEQ_LEN / self.SEQ_STRIDE), 2)

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

                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

                peak_batch = peak_batch.view((1, self.batch_size, 1)).float().to(device)       # (seq_len, batch, input_size)
                l_dna_batch = l_dna_batch.transpose(0, 2).transpose(1, 2).float().to(device)   # (seq_len, batch, input_size)
                t_dna_batch = t_dna_batch.transpose(0, 2).transpose(1, 2).float().to(device)   # (seq_len, batch, input_size)

                batches.append((l_dna_batch, t_dna_batch, peak_batch))

        # print(time.clock() - start)

        return batches
