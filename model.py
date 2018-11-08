# Torch imports
import torch
import torch.nn as nn
from torch.autograd import Variable as var

import numpy as np

class BSPredictor(nn.Module):

    ONE_HOT_DIM = 4
    RECURRENT_LAYERS = 2
    DROPOUT = 0.9

    def __init__(self, hidden_size, rnn_type=nn.GRU):
        super(BSPredictor, self).__init__()
        self.hidden_size = hidden_size

        # Recurrent layers
        self.rl = rnn_type(input_size=self.ONE_HOT_DIM, hidden_size=hidden_size, num_layers=self.RECURRENT_LAYERS, dropout=self.DROPOUT)
        self.rlr = rnn_type(input_size=self.ONE_HOT_DIM, hidden_size=hidden_size, num_layers=self.RECURRENT_LAYERS, dropout=self.DROPOUT)

        # Final feed-forward layer
        self.ffl = nn.Linear(hidden_size * 2, 1)

        self.reset_hidden_states()


    def reset_hidden_states(self, batch_size=1, for_batch=None):
        if for_batch is not None:
            batch_size = for_batch.shape[1]

        # Initialize recurrent hidden states
        self.rl_h = self.init_hidden(batch_size=batch_size)
        self.rlr_h = self.init_hidden(batch_size=batch_size)

        if for_batch is not None:
            device = for_batch.device
            self.rl_h = self.rl_h.to(device)
            self.rlr_h = self.rlr_h.to(device)


    def init_hidden(self, batch_size=1):
        return var(torch.zeros(self.RECURRENT_LAYERS, batch_size, self.hidden_size))


    def forward(self, leading_dna, trailing_dna):
        self.reset_hidden_states(for_batch=leading_dna)

        rl_out, self.rl_h = self.rl(leading_dna, self.rl_h)
        rlr_out, self.rlr_h = self.rlr(self.flip_input(trailing_dna), self.rlr_h)

        r_out = torch.cat((self.rl_h[1], self.rlr_h[1]), 1)

        prediction_lin = self.ffl(r_out)
        prediction = torch.sigmoid(prediction_lin)

        return prediction


    def flip_input(self, input):
        flipped_array = np.flip(input.data.numpy(), 0).copy()
        return var(torch.from_numpy(flipped_array))