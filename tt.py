import torch

# Linear regression
# f = w * x  + b


# here : f = 2 * x1 + 3 * x2

X = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], 
                 [2, 4, 6, 8, 10, 12, 14, 16]],
                 dtype=torch.float32)
# Y = 2 * X[0] + 3 * X[1]
Y = torch.tensor([8, 16, 24, 32, 40, 48, 56, 64], dtype=torch.float32)

#Y = torch.tensor([8, 16, 6, 8, 10, 12, 14, 16], dtype=torch.float32)

#w = torch.tensor([6.0, 4.0, 7.0], dtype=torch.float32, requires_grad=True)
#w = torch.randint(0, 5, (3,1), dtype=torch.float32, requires_grad=True)

# model output
def forward(x):
    return w[0] * x[0] + w[1] * x[1] + w[2]

# loss = MSE
def loss(x, y, y_pred):
    y = 2*x[0] + 3*x[1] + 4
    yy = 1.6*x[0] + 3.2*x[1]
    """print("y: ", y)
    print("yy: ", yy)
    print("x[0]:", x[0])
    print("x[1]:", x[1])"""
    ssqe = (y_pred - y)**2
    mssqe = ssqe.mean()
    """
    print("ssqe: ", ssqe)
    print("mssqe: ", mssqe)
    """
    return mssqe/10


X_test = [5.0, 10.0]

#print(f'Prediction before training: f({X_test}) = {forward([5.0, 10.0]]):.3f}')

# Training
learning_rate = 0.07
n_epochs = 2000

def train(X, Y, w, epochs):
    for epoch in range(n_epochs):
        # predict = forward pass
        Y_pred = forward(X)
        #print("\nY_pred: ", Y_pred)

        # loss
        l = loss(X, Y, Y_pred)

        # calculate gradients = backward pass
        l.backward()

        # update weights
        #w.data = w.data - learning_rate * w.grad
        with torch.no_grad():
            w -= learning_rate * w.grad
        
        # zero the gradients after updating
        w.grad.zero_()

        if (epoch+1) % 2000 == 0:
            #print("\nY_pred: ", Y_pred)
            print(f'epoch {epoch+1}: w0 = {w[0].item():.3f}, w1 = {w[1].item():.3f}, w2 = {w[2].item():.3f}, loss = {l.item():.3f}')
        continue

for i in range(10):
    w = torch.randint(0, 5, (3,1), dtype=torch.float32, requires_grad=True)
    print(f'\nw0 = {w[0].item():.3f}, w1 = {w[1].item():.3f}, w2 = {w[2].item():.3f}')
    train(X, Y, w, n_epochs)

print(f'Prediction after training: f({X_test}) = {forward(X_test).item():.3f}')