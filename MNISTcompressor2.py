import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
input_size = 784 # 28x28
hidden_size = 784
num_classes = 10
num_epochs = 50
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
"""
examples = iter(test_loader)
example_data, example_targets = next(examples)  # iter error here
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(example_data[i][0], cmap='gray')
# plt.show()
"""
# Fully connected neural network with one hidden layer with the output layer being the same size as input
class NeuralNet2(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet2, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        #self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, input_size)  
    
    def forward(self, x):
        out = self.l1(x)
        #out = self.relu(out)
        out = self.l2(out)
        # no activation and no softmax at the end
        return out
model = NeuralNet2(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# Train the model 
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        # origin shape: [100, 1, 28, 28]
        # resized: [100, 784]
        # print images.type
        #print(images.type())
        #print(images.shape)
        images = images.reshape(-1, 28*28).to(device)
        noise = np.random.normal(0, 0.1, 100*784)
        # convert to tensor
        noise = torch.from_numpy(noise).float()
        images = images + noise.reshape(100, 784)
        labels = labels.to(device)
        
        # Forward pass and loss calculation
        outputs = model(images)
        loss = criterion(outputs, images)
        images_saved = images
        outputs_saved = outputs
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if (i+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

oo = outputs_saved.detach().numpy()
oo = np.reshape(oo, (100, 1, 28, 28))
ii = images_saved.detach().numpy()
ii = np.reshape(ii, (100, 1, 28, 28))

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

        outputs = model(images)
        loss = criterion(outputs, images)

    acc = loss / n_samples
    print(f'Subtracted loss on the {n_samples} test images: {acc:.5f} %')
