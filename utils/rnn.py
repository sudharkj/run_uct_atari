import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self, name):
        super(RNN, self).__init__()
        self.name = name
        # TODO modify the layers
        # self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        # self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        # self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        # self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.fc1 = nn.Linear(64 * 8 * 8, 512)
        # self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # TODO modify the layers
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        # x = self.pool(x)
        # x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x))
        # x = self.pool(x)
        # # x = x.view(-1, num_flat_features(x))
        # x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        return x

