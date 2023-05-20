import torch
import numpy as np

# Create a torch tensor [1, 2, 3]
x = torch.tensor([1, 2, 3])
# Create a numpy array [4, 5, 6]
y = np.array([4, 5, 6])
# Add them together
z = x + y
print("z:", z)
# Print the data type of z
print("z's data type:", z.dtype)
