# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as var

import numpy as np

class BSPredictor(nn.Module):

    ONE_HOT_DIM = 4
    RECURRENT_LAYERS = 2

    def __init__(self, hidden_size, rnn_type=nn.GRU):
        super(BSPredictor, self).__init__()
        self.hidden_size = hidden_size

        # Recurrent layers
        # TODO: Introduce dropout?
        self.rl = rnn_type(input_size=self.ONE_HOT_DIM, hidden_size=hidden_size, num_layers=self.RECURRENT_LAYERS)
        self.rlr = rnn_type(input_size=self.ONE_HOT_DIM, hidden_size=hidden_size, num_layers=self.RECURRENT_LAYERS)

        # Final feed-forward layer
        self.ffl = nn.Linear(hidden_size * 2, 1)

        # Initialize recurrent hidden states
        self.rl_h = self.init_hidden()
        self.rlr_h = self.init_hidden()


    def init_hidden(self, batch_size=1):
        return var(torch.zeros(self.RECURRENT_LAYERS, batch_size, self.hidden_size))


    def forward(self, nucleotides):
        rl_out, self.rl_h = self.rl(nucleotides, self.rl_h)
        rlr_out, self.rlr_h = self.rlr(self.flip_input(nucleotides), self.rlr_h)

        r_out = torch.cat((self.rl_h[1], self.rlr_h[1]), 1)

        prediction_lin = self.ffl(r_out)
        prediction = F.sigmoid(prediction_lin)

        return prediction


    def flip_input(self, input):
        flipped_array = np.flip(input.data.numpy(), 0).copy()
        return var(torch.from_numpy(flipped_array))