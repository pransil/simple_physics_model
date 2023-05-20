# data_loader.py

# data.txt is a space delimited file with x1 y1 x2 y2 x3 y3 vx vy g
# where the xy pairs are the positions of an object in ballistic flight
# with vx, vy being the initial velocity and g being the acceleration due to gravity
# The goal is to predict x3, y3 given x1, y1, x2, y2, vx, vy, g

import numpy as np
import torch
from sklearn.model_selection import train_test_split

def load_data(data_file, print_flag=False):
    """Load data from data_file into train (80%) and test (20%) pytorch tensors
    
    Args: data_file (str): path to data file
            print_flag (bool): print data shapes if True
    Returns: trainX (torch.tensor): training inputs
             trainY (torch.tensor): training outputs
             testX (torch.tensor): test inputs
             testY (torch.tensor): test outputs
    """
    data = np.loadtxt('data.txt', delimiter=' ')
    data = data.astype(np.float32)
    # Split data 80/20 into train and test sets
    train, test = train_test_split(data, test_size=0.2)
    train = torch.from_numpy(train)
    test = torch.from_numpy(test)
    # Remove x3, y3 from inputs, put them into trainY
    trainX = torch.cat((train[:, :4], train[:, 6:]), 1)
    trainY = train[:, 4:6]
    testX = torch.cat((test[:, :4], test[:, 6:]), 1)
    testY = test[:, 4:6]
    if print_flag==True:
        print("Training input data: (trainX)", trainX.shape)
        print("Training output data: (trainY)", trainY.shape)
        print("Test input data: (testX)", testX.shape)
        print("Test output data: (testY)", testY.shape)

    return trainX, trainY, testX, testY
