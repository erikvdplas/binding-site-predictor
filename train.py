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
import random
import functools


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
                      help='Determines after how many batches the model will be saved.', default=1000)
    parser.add_option("--batch-size", action='store', dest='batch_size', type='int',
                      help='Determines batch size.', default=16)
    (params, _) = parser.parse_args(sys.argv)

    if params.needs_seed:
        print('Seeding random')
        random.seed(1)
        torch.manual_seed(1)

    model = BSPredictor(hidden_size=128)

    if os.path.exists(SAVE_PATH):
        print('Loading pre-trained model')
        model.load_state_dict(torch.load(SAVE_PATH))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    loss_function = LogLoss()

    if torch.cuda.device_count() > 1:
        print("Using %d GPUs for data parallelism" % torch.cuda.device_count())
        model = nn.DataParallel(model, dim=1)
        loss_function = nn.DataParallel(loss_function, dim=1)
    elif torch.cuda.device_count() == 1:
        model.to(device)
        loss_function.to(device)

    optimizer = optim.Adam(model.parameters())

    test_part = 0.2
    dataset = data.BSDataset(params.chip_path, params.batch_size, test_part=test_part)

    # batch_size * adjacent_len * first_num_underneath should be >>1000, so ATM 5 should be 2^(8-10)
    epoch_size = 512 * int(1 / test_part)

    training_iter = 0

    for epoch in range(params.epochs):
        print('Training epoch %d' % epoch)

        train_losses = []

        for train_batch in dataset.load_batches(int(epoch_size * (1 - test_part))):
            if training_iter % params.save_interval == 0:
                print('Saving model at epoch %d \t training iter %d' % (epoch, training_iter))
                torch.save(model.state_dict(), SAVE_PATH)

            model.zero_grad()

            (leading_input, trailing_input, target) = train_batch
            (leading_input, trailing_input, target) = (leading_input.to(device), trailing_input.to(device), target.to(device))
            prediction = model(var(leading_input), var(trailing_input))

            loss = loss_function(prediction, var(target))
            train_losses.append(loss.data.item())

            loss.backward()
            optimizer.step()

            training_iter += 1

        average_loss = float(functools.reduce(lambda x, y: x + y, train_losses)) / float(len(train_losses))
        print('Average train loss: \t %f' % average_loss)
        with open('train-loss.log', 'a') as f:
            f.write(str(average_loss) + '\n')

        with torch.no_grad():
            test_losses = []

            for test_batch in dataset.load_batches(int(epoch_size * test_part), test=True):
                (leading_input, trailing_input, target) = test_batch
                (leading_input, trailing_input, target) = (leading_input.to(device), trailing_input.to(device), target.to(device))
                prediction = model(var(leading_input), var(trailing_input))

                loss = loss_function(prediction, var(target))
                test_losses.append(loss.data.item())

            average_loss = float(functools.reduce(lambda x,y: x + y, test_losses)) / float(len(test_losses))

            print('Average test loss: \t %f' % average_loss)

            with open('test-loss.log', 'a') as f:
                f.write(str(average_loss) + '\n')

            dataset.test_offset = 0
            dataset.test_arr_offset = 0

