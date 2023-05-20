# model2.py - define the model
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """A simple linear model with 2 hidden layers"""
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(7, 3)
        self.fc2 = nn.Linear(3, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        #print("Outputs:", x.shape)
        #print("Outputs:", x)
        return x
    
