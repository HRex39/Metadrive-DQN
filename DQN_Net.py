'''
MIT License
Copyright (c) 2022 HuangChenrui<hcr2077@outlook.com>
'''

# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# Net Params
N_ACTIONS = 3*3 # Action Space is an array[steering, acceleration] like [-0.5,0.5] which need to be discrete
N_STATES = 80*80*4

class DQN_Net(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=(8,8), stride=4, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2), stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4,4), stride=2, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2), stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=(2,2), stride=2, padding=1)
        self.fc1   = nn.Linear(in_features=256, out_features=256)
        self.fc2   = nn.Linear(in_features=256, out_features=N_ACTIONS)

        self.N_ACTIONS = N_ACTIONS
        self.N_STATES = N_STATES

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)  # Maybe the problem is here
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        actions_value = x
        return actions_value
