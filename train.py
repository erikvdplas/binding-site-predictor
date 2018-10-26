# Torch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable as var

# Project imports
from model import BSPredictor

# Utility imports
import optparse
import sys
import os
import data


SAVE_PATH = 'bspredictor.model'

class LogLoss(nn.Module):

    def __init__(self):
        super(LogLoss, self).__init__()

    def forward(self, x, y):
        losses = y * torch.log(x) + (1 - y) * torch.log(1 - x)
        return -torch.mean(losses)


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option("--seed", action='store_true', dest='needs_seed',
                      help='Determines wheter PyTorch will be seeded.')
    parser.add_option("--epochs", action='store', dest='epochs', type='int',
                      help='Determines how many epochs of training will be executed.')
    parser.add_option("--chip-path", action='store', dest='chip_path', type='str',
                      help='Determines location of pickled ChIP data file.')
    parser.add_option("--save-interval", action='store', dest='save_interval', type='int',
                      help='Determines after how many batches the model will be saved.', default=50)
    (params, _) = parser.parse_args(sys.argv)

    if params.needs_seed:
        torch.manual_seed(1)

    model = BSPredictor(hidden_size=128)

    if os.path.exists(SAVE_PATH):
        print('Loading pre-trained model')
        model.load_state_dict(torch.load(SAVE_PATH))

    loss_function = LogLoss()
    optimizer = optim.Adam(model.parameters())

    batches = data.load_batches(params.chip_path, 8, 2)

    for epoch in range(params.epochs):
        print('Training epoch %d' % epoch)

        for batch_idx, (input, target) in enumerate(batches): # Pseudocode
            if (epoch * len(batches) + batch_idx) % params.save_interval == 0:
                print('Saving model at epoch %d \t batch %d' % (epoch, batch_idx))
                torch.save(model.state_dict(), SAVE_PATH)

            model.zero_grad()

            prediction = model(var(input))

            loss = loss_function(prediction, var(target))
            print(loss.data)
            loss.backward(retain_graph=True)
            optimizer.step()