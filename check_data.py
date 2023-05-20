# Check that the data is as expected
import numpy as np

# Load data.txt into an np array
data = np.loadtxt('data.txt', delimiter=' ')
# Print the shape of data
print("Data shape:", data.shape)
# Print the first 5 rows of all columns of data with 3 decimal places
np.set_printoptions(precision=3)
print("Data:", data[:5, :])
