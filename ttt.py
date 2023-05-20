#ttt.py - testing simple models
import numpy as np
import torch
import sys
from sys import exit

# Linear regression
# Y1 = 2 * x0 + 3 * x2 + 4
# Y2 = x0**2 + 2*x1 + 2

X = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], 
                 [2, 4, 6, 8, 10, 12, 14, 16]],
                 dtype=torch.float32)

#Y = torch.zeros(2, 8, requires_grad=True)
Y = np.zeros((2, 8))
Y[0] = 2 * X[0] + 3 * X[1] + 4
Y[1] = X[0]**2 + 2*X[1] + 2
Y = torch.tensor(Y, dtype=torch.float32, requires_grad=True)
print("X: ", X)
print("Y: ", Y)

# model output
def forward(x):
    Y_pred = torch.zeros(2, 8)
    Y_pred[0] = w[0] * x[0] + w[1] * x[1] + w[2]
    Y_pred[1] = w[3] * x[0]**2 + w[4] * x[1] + w[5]
    return Y_pred

# loss = MSE
def loss(y, y_pred):
    ssqe = (y_pred - y)**2
    mssqe = torch.mean(ssqe, 1)
    print("y_pred: ", y_pred)
    print("ssqe: ", ssqe)
    print("mssqe: ", mssqe)
    print("Y.size()[1]: ", Y.size()[1])
    return mssqe/Y.size()[1]
'''
def loss1(y, y_pred):
    ssqe = (y_pred - y)**2
    mssqe = ssqe.mean()
    """
    print("ssqe: ", ssqe)
    print("mssqe: ", mssqe)
    """
    return mssqe/10
'''
X_test = [5.0, 10.0]

#print(f'Prediction before training: f({X_test}) = {forward([5.0, 10.0]]):.3f}')

# Training
learning_rate = 0.01
n_epochs = 2000

def train(X, Y, w, epochs):
    for epoch in range(n_epochs):
        Y_pred = forward(X)
        #print("\nY_pred: ", Y_pred)

        l = loss(Y, Y_pred)
        l.backward()
        # update weights
        with torch.no_grad():
            w -= learning_rate * w.grad
        
        w.grad.zero_()
        if (epoch+1) % 2000 == 0:
            print(f'epoch {epoch+1}: w0 = {w[0].item():.3f}, w1 = {w[1].item():.3f}, w2 = {w[2].item():.3f}, loss = {l.item():.3f}')
        continue

for i in range(10):
    w = torch.randint(0, 5, (6,1), dtype=torch.float32, requires_grad=True)
    print(f'\nw0 = {w[0].item():.3f}, w1 = {w[1].item():.3f}, w2 = {w[2].item():.3f}')
    train(X, Y, w, n_epochs)

print(f'Prediction after training: f({X_test}) = {forward(X_test).item():.3f}')