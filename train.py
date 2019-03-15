import argparse
import datetime
import os

from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils.rnn import RNN
from utils.runs import load_runs


parser = argparse.ArgumentParser(description="Run commands",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('dirs', type=str, nargs='+', help="Directories with uct data")
parser.add_argument('--in_model', type=str, default=None, help="Saved model filename")
parser.add_argument('--out_model', type=str, default='models/model_%s' % datetime.now().strftime("%Y%m%d-%H%M%S"),
                    help="Filename for trained model")
parser.add_argument('--save_period', type=int, default=10, help="Interval between checkpoints")
parser.add_argument('--network', type=str, default='cnn', help="Network architecture")
parser.add_argument('--n_frames', type=int, default=2, help="Number of frames to stack")
parser.add_argument('--width', type=int, default=84, help="Width of frame")
parser.add_argument('--height', type=int, default=84, help="Height of frame")
parser.add_argument('--downsample', type=float, default=None, help="Factor of downsampling image")
parser.add_argument('--loss', type=str, default='cross_entropy', help="Type of loss function: [cross_entropy]")
parser.add_argument('--batch', type=int, default=32, help="Number of samples per batch")
parser.add_argument('--samples_per_epoch', type=int, default=1000, help="Number of samples per epoch")
parser.add_argument('--epochs', type=int, default=100, help="Number of epochs")
parser.add_argument('--augment', action='store_true', help="Augment images")
parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate")
parser.add_argument('--weight_runs', action='store_true', help="Weight runs according to reward obtained in run")
parser.add_argument('--norm', action='store_true', help="Do value normalization per state or not")
parser.add_argument('--norm_coeff', type=float, default=1, help="Normalization coefficient")
parser.add_argument('--entropy', type=float, default=0.001, help="Entropy coefficient for policy loss")
parser.add_argument('--flip', action='store_true', help="Flip image and action vertically")
parser.add_argument('--color', action='store_true', help="Process color images instead of grayscale")
parser.add_argument('--min_run_score', type=float, default=None, help="Minimum score in run to process run")
parser.add_argument('--generator_workers', type=int, default=1, help="Number of workers to generate data.")


def train(net, optimizer, criterion, data):
    # TODO have to match the inputs and label format
    inputs, labels = data
    inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
    net.train()
    optimizer.zero_grad()
    outputs = net(inputs)  # forward
    loss = criterion(outputs, labels)  # loss
    loss.backward()  # accumulated gradient
    optimizer.step()


if __name__ == "__main__":
    args = parser.parse_args()
    # constant values
    folders = {
        "results": "results",
        "models": "models"
    }
    dirs = ""
    kwargs = {
        'height': 50,
        'width': 50,
        'downsample': None,
        'min_score': np.inf
    }

    # generate required folders
    for key in folders.keys():
        try:
            os.makedirs(folders[key])
        except FileExistsError:
            # if file exists, pass
            pass

    x_train = load_runs(dirs, **kwargs)
    if x_train is not None and len(x_train) > 0:
        rnn = RNN("Initial Model")
        adam = optim.Adam(rnn.parameters(), lr=0.01)
        cross_entropy_loss = nn.CrossEntropyLoss()
        train(rnn, adam, cross_entropy_loss, x_train)
        torch.save(rnn.state_dict(), './%s/training-%s.pth' % (folders['models'], 'rnn'))
