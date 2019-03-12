import os

from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .rnn import RNN
from .runs import load_runs


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
