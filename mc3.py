import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt

# Hyper-parameters 
input_size = 784 # 28x28
num_epochs = 1
batch_size = 100
learning_rate = 0.001

# MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', 
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

torch.set_printoptions(sci_mode=False)

# Fully connected neural network with one hidden layer with the output layer being the same size as input
class NeuralNet2(nn.Module):
    def __init__(self, input_size):
        super(NeuralNet2, self).__init__()
        self.l1 = nn.Linear(input_size, input_size) 
    
    def forward(self, x):
        out = self.l1(x)
        # no activation and no softmax at the end
        return out

class NeuralNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_size))
        
    def forward(self, x):
        return (x @ self.weight)
    
model = NeuralNet2(input_size)

def lss(y, y_pred):
    z = y_pred - y
    # z requires grad
    z.requires_grad_()
    return z

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

optimizer = torch.optim.Adam(model.parameters(), learning_rate)

# Train the model 
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        images = images.reshape(-1, 28*28)
        noise = np.random.normal(0, 0.1, batch_size*784)
        noise = torch.from_numpy(noise).float()
        images = images + noise.reshape(batch_size, 784)
        # Forward pass and loss calculation
        outputs = model(images)
        images_saved = images
        outputs_saved = outputs
        #l = loss(outputs, images).requires_grad_()
        #l = torch.nn.MSELoss(reduction='None')

        #loss = criterion(outputs, images)
        loss = lss(outputs, images)
        print('loss.shape: ', loss.shape)
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        #print('l: ', l)

        #l.backward()
        #optimizer.step()
        #optimizer.zero_grad()

        if (i+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
            print('model.params: ', model.parameters())
            input("press any key to continue")

oo = torch.reshape(outputs_saved, (batch_size, 1, 28, 28))
ii = torch.reshape(images_saved, (batch_size, 1, 28, 28))

for i in range(1, 4):
    plt.subplot(3, 3,i)
    plt.imshow(ii[i][0], cmap='gray')
for i in range(1, 4):
    plt.subplot(3, 3,i+3)
    plt.imshow(oo[i][0], cmap='gray')
ss = np.abs(ii - oo)
for i in range(1, 4):
    plt.subplot(3, 3,i+6)
    plt.imshow(ss[i][0], cmap='gray')
plt.show()

"""
outputs = outputs_saved.detach().numpy()
outputs = np.reshape(outputs, (100, 1, 28, 28))

images = images_saved.detach().numpy()
images = np.reshape(images, (100, 1, 28, 28))
print('outputs_saved shape: ', outputs_saved.size())
print('images_saved shape: ', images_saved.shape)

subtracted_outputs = np.abs(images - outputs)
for j, (images, labels) in enumerate(train_loader): 
    break
for i in range(1, 4):
    plt.subplot(2, 3,i)
    plt.imshow(images[i][0], cmap='gray')
for i in range(4, 7):
    plt.subplot(2, 3,i)
    plt.imshow(outputs[i][0], cmap='gray')

for i in range(6):
    #plt.subplot(2,3,i+4)
    plt.imshow(outputs[i][0], cmap='gray')

for i in range(6):
    #plt.subplot(2,3,i+7)
    plt.imshow(subtracted_outputs[i][0], cmap='gray')

plt.show()
"""

# Test the model: we don't need to compute gradients
with torch.no_grad():
    n_correct = 0
    n_samples = len(test_loader.dataset)

    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)

        outputs = forward(images)
        loss = criterion(outputs, images)

    acc = loss / n_samples
    print(f'Subtracted loss on the {n_samples} test images: {acc:.5f} %')
    print("w: ", w[:10])
