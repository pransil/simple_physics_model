# main.py - train and test a simple PyTorch regression model

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import data_loader
import model2

# Load data
trainX, trainY, testX, testY = data_loader.load_data('data.txt', print_flag=True)
# Define and setup the model
net = model2.Net()
net = net.float()
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)
# Track the loss history
loss_history = []

# Train the model
for epoch in range(10):
    running_loss = 0.0
    for i, X, Y in enumerate(trainX, trainY):
        optimizer.zero_grad()
        Y_pred = net(X)
        loss = criterion(Y_pred, Y)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

       # Print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # Print every 100th mini-batch
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished Training')

# Test the model
val_losses = []

with torch.no_grad():
    for i, X, Y in enumerate(testX, testY):
        Y_pred = net(X)
        #print("Outputs shape:", outputs.shape)
        #print("Outputs:", outputs)
        loss = criterion(Y_pred, Y)
        val_losses.append(loss.item())
        running_loss += loss.item()
 
        if i % 100 == 99:    # Print every 100th mini-batch
            print('%d loss: %.3f' % (i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished Testing')

 
plt.figure(figsize=(10,5))
plt.title("Training and Validation Loss")
plt.plot(val_losses,label="val")
plt.plot(loss_history,label="train")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()    

# Save the model
PATH = './ballistic_net.pth'
torch.save(net.state_dict(), PATH)


    


